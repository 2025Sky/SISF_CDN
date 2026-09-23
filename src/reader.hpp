/*

SISF CDN - Scalable Image Stack Format Content Delivery Network
Copyright 2023-2026 by the Regents of the University of Michigan
Licensed under the terms specified in LICENSE.md

@author: Logan A Walker, PhD <loganaw@umich.edu>

*/

#include <glob.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>
#include <string>
#include <algorithm>
#include <queue>
#include <deque>
#include <mutex>
#include <atomic>
#include <map>
#include <unordered_map>
#include <utility>
#include <chrono>

#include <nlohmann/json.hpp>

#include "vidlib.hpp"

#include "../zstd/lib/zstd.h"

#include "absl/flags/flag.h"
#include "absl/flags/marshalling.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include <half.hpp>

#include "tensorstore/array.h"
#include "tensorstore/context.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "examples/data_type_invoker.h"
#include "absl/flags/parse.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/open.h"
#include "tensorstore/read_write_options.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/progress.h"
#include "tensorstore/rank.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
#include "tensorstore/util/utf8_string.h"

#include <sys/stat.h>

#define CHUNK_TIMER 0
#define DEBUG_SLICING 0
#define IO_RETRY_COUNT 5

size_t mchunk_uuid = 0;
std::mutex mchunk_uuid_mutex;

struct global_chunk_line
{
    size_t mchunk;
    size_t chunk;
    uint16_t *ptr;
    // Extent the buffer was decoded with. A reload can change a chunk's
    // extent, so a line for another extent is a miss.
    size_t sizex, sizey, sizez;
};

enum ArchiveType
{
    SISF_JSON,
    SISF,
    ZARR,
    DESCRIPTOR
};

using json = nlohmann::json;

std::chrono::duration cache_lock_timeout = std::chrono::milliseconds(10);

// A whole decimal number (digits only, no sign, no spaces), as the
// environment settings take them. False for anything else, or a number
// that does not fit in a size_t.
bool parse_decimal(const std::string &s, size_t &out)
{
    if (s.empty() || s.size() > 19) // 19 digits always fit in 64 bits
    {
        return false;
    }
    size_t v = 0;
    for (char ch : s)
    {
        if (ch < '0' || ch > '9')
        {
            return false;
        }
        v = v * 10 + (ch - '0');
    }
    out = v;
    return true;
}

// The global chunk cache: a ring of decoded chunks, the oldest replaced
// first, with a hash index from (mchunk, chunk) to the chunk's line, so a
// lookup no longer scans every line. Its size is CHUNK_CACHE_LINES (100
// when unset). Everything below is guarded by global_chunk_cache_mutex.
const size_t global_cache_default_lines = 100;
const size_t global_cache_max_lines = 1000000;

size_t chunk_cache_lines_from_env()
{
    const char *s = std::getenv("CHUNK_CACHE_LINES");
    if (s == nullptr || *s == '\0')
    {
        return global_cache_default_lines;
    }
    size_t n = 0;
    if (!parse_decimal(s, n) || n == 0 || n > global_cache_max_lines)
    {
        std::cerr << "CHUNK_CACHE_LINES ignored (not a whole number from 1 to " << global_cache_max_lines
                  << "): " << s << "; using " << global_cache_default_lines << std::endl;
        return global_cache_default_lines;
    }
    return n;
}

struct global_chunk_key_hash
{
    size_t operator()(const std::pair<size_t, size_t> &k) const
    {
        return std::hash<size_t>()(k.first * 0x9E3779B97F4A7C15ULL ^ k.second);
    }
};

std::timed_mutex global_chunk_cache_mutex;
size_t global_cache_size = chunk_cache_lines_from_env();
global_chunk_line *global_chunk_cache = (global_chunk_line *)calloc(global_cache_size, sizeof(global_chunk_line));
size_t global_chunk_cache_last = 0;
// Only lines that hold a chunk (ptr != 0) are indexed, one line per chunk
std::unordered_map<std::pair<size_t, size_t>, size_t, global_chunk_key_hash> global_chunk_index = []()
{
    std::unordered_map<std::pair<size_t, size_t>, size_t, global_chunk_key_hash> m;
    m.reserve(global_cache_size);
    return m;
}();

// The line holding (mchunk, chunk), or nullptr
global_chunk_line *global_cache_find(size_t mchunk, size_t chunk)
{
    auto it = global_chunk_index.find({mchunk, chunk});
    return it == global_chunk_index.end() ? nullptr : global_chunk_cache + it->second;
}

// Frees the chunk a line holds, if any, and removes it from the index
void global_cache_drop(size_t slot)
{
    global_chunk_line &line = global_chunk_cache[slot];
    if (line.ptr == 0)
    {
        return;
    }
    free(line.ptr);
    line.ptr = 0;
    auto it = global_chunk_index.find({line.mchunk, line.chunk});
    if (it != global_chunk_index.end() && it->second == slot)
    {
        global_chunk_index.erase(it);
    }
}

// Takes ownership of ptr. A line the chunk already has (e.g. for an older
// extent) is dropped first, so a lookup can only find the newest one.
void global_cache_insert(size_t mchunk, size_t chunk, uint16_t *ptr, size_t sizex, size_t sizey, size_t sizez)
{
    global_chunk_line *old = global_cache_find(mchunk, chunk);
    if (old != nullptr)
    {
        global_cache_drop(old - global_chunk_cache);
    }

    const size_t slot = global_chunk_cache_last;
    global_cache_drop(slot);
    global_chunk_line &line = global_chunk_cache[slot];
    line.mchunk = mchunk;
    line.chunk = chunk;
    line.ptr = ptr;
    line.sizex = sizex;
    line.sizey = sizey;
    line.sizez = sizez;
    global_chunk_cache_last = (slot + 1) % global_cache_size;

    // Last: if this throws, the line still owns ptr and is freed when its slot is reused
    global_chunk_index[{mchunk, chunk}] = slot;
}

// The log lines this fork adds are rate limited: at most one line per second
// per kind, and the next line that gets through says how many were dropped
// in between. A failure that persists (a corrupt chunk, a stale dataset) is
// hit on every request and would otherwise fill the log. Atomics only, so
// logging never waits on a lock.
class log_limiter
{
    std::atomic<int64_t> next_ms{0};
    std::atomic<uint64_t> dropped{0};

public:
    // True when the caller may write its line now; note is then empty or
    // names how many lines of this kind were dropped since the last one.
    bool allow(std::string &note)
    {
        const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count();
        int64_t next = next_ms.load(std::memory_order_relaxed);
        if (now_ms < next || !next_ms.compare_exchange_strong(next, now_ms + 1000, std::memory_order_relaxed))
        {
            dropped.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        const uint64_t n = dropped.exchange(0, std::memory_order_relaxed);
        note = n == 0 ? "" : " (" + std::to_string(n) + " similar lines suppressed)";
        return true;
    }
};

log_limiter log_limit_chunk_oom;
log_limiter log_limit_chunk_decode;
log_limiter log_limit_entry_write;
log_limiter log_limit_write_oom;
log_limiter log_limit_append;
log_limiter log_limit_header;
log_limiter log_limit_read_outside;
log_limiter log_limit_stale_archive;
log_limiter log_limit_stale_extent;
log_limiter log_limit_read_refused;
log_limiter log_limit_write_refused;
log_limiter log_limit_write_failed;
log_limiter log_limit_archive_meta;

// https://stackoverflow.com/questions/8401777/simple-glob-in-c-on-unix-system
std::vector<std::string> glob_tool(const std::string &pattern)
{
    std::vector<std::string> filenames;

    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // do the glob operation
    const int glob_flag = GLOB_NOSORT | GLOB_TILDE;
    int return_value = glob(pattern.c_str(), glob_flag, NULL, &glob_result);

    if (return_value == GLOB_NOMATCH)
    {
        // Return nothing if no results found
        globfree(&glob_result);
        return filenames;
    }

    if (return_value != 0)
    {
        // Fail if another error code is found
        globfree(&glob_result);
        std::stringstream ss;
        ss << "glob() failed with return_value " << return_value << std::endl;
        throw std::runtime_error(ss.str());
    }

    // collect all the filenames into a std::list<std::string>
    for (size_t i = 0; i < glob_result.gl_pathc; ++i)
    {
        filenames.push_back(std::string(glob_result.gl_pathv[i]));
    }

    std::sort(filenames.begin(), filenames.end());

    globfree(&glob_result);
    return filenames;
}

// Return time_t of given files modification time,
// 0 if file does not exist or error occurs
time_t get_file_mtime(std::string filename)
{
    struct stat result;
    if (stat(filename.c_str(), &result) == 0)
    {
        return result.st_mtime;
    }
    return 0;
}

// Modification time in nanoseconds, 0 if the file does not exist. A
// re-conversion can rewrite a file within the second it was first read.
int64_t get_file_mtime_ns(const std::string &filename)
{
    struct stat result;
    if (stat(filename.c_str(), &result) == 0)
    {
        return (int64_t)result.st_mtim.tv_sec * 1000000000 + result.st_mtim.tv_nsec;
    }
    return 0;
}

struct metadata_entry
{
    uint64_t offset;
    uint32_t size;
};

class packed_reader
{
private:
    const size_t entry_file_line_size = 8 + 4;
    const size_t header_size_expected = sizeof(uint16_t) * 7 + sizeof(uint64_t) * 9;

public:
    bool is_valid;
    uint16_t channel_count;
    uint16_t dtype, version;
    uint16_t compression_type;
    uint16_t chunkx, chunky, chunkz;
    uint64_t sizex, sizey, sizez;
    uint64_t countx, county, countz;

    uint64_t cropstartx, cropstarty, cropstartz;
    uint64_t cropendx, cropendy, cropendz;

    size_t max_chunk_size;
    size_t header_size;

    size_t this_mchunk_id;

    // In nanoseconds: a rewrite that finishes within the second of the last
    // check (e.g. after a reload that found the header cut short) must still
    // count as a change
    int64_t last_meta_read_time = 0;

    std::string meta_fname, data_fname;

    // Bumped, under global_chunk_cache_mutex, after a chunk of this mchunk is
    // rewritten and when its cache lines are dropped. load_chunk samples it
    // before reading a chunk's entry and puts its decode in the global cache
    // only if it has not moved: a decode begun before a write finished may be
    // of the old chunk, and the write's invalidation has already run.
    std::atomic<uint64_t> data_gen{0};

    packed_reader(size_t chunk_id, std::string metadata_fname_in, std::string data_fname_in)
    {
        is_valid = false;
        this_mchunk_id = chunk_id;
        meta_fname = metadata_fname_in;
        data_fname = data_fname_in;

        if (last_meta_read_time == 0)
        {
            check_mtime_hasmodified();
        }

        reload_metadata(false);
    }

    ~packed_reader()
    {
    }

    bool check_mtime_hasmodified()
    {
        int64_t mtime = get_file_mtime_ns(meta_fname);

        if (last_meta_read_time != mtime)
        {
            last_meta_read_time = mtime;
            return true;
        }

        return false;
    }

    void clear_cache_lines()
    {
        std::lock_guard<std::timed_mutex> lock(global_chunk_cache_mutex);
        // A decode made under the header being replaced must not come back
        data_gen.fetch_add(1);
        for (size_t i = 0; i < global_cache_size; i++)
        {
            if (global_chunk_cache[i].mchunk == this_mchunk_id)
            {
                global_cache_drop(i);
            }
        }
    }

    void reload_metadata(bool reset_cache = true)
    {
        std::ifstream file(meta_fname, std::ios::in | std::ios::binary);

        if (file.fail())
        {
            std::cerr << "Fopen failed (chunk metadata)" << std::endl;
            return;
        }

        if (last_meta_read_time == 0)
        {
            check_mtime_hasmodified();
        }

        std::streamsize bytes_read = 0;
        file.read((char *)&version, sizeof(uint16_t));
        bytes_read += file.gcount();
        file.read((char *)&dtype, sizeof(uint16_t));
        bytes_read += file.gcount();
        file.read((char *)&channel_count, sizeof(uint16_t));
        bytes_read += file.gcount();
        file.read((char *)&compression_type, sizeof(uint16_t));
        bytes_read += file.gcount();

        file.read((char *)&chunkx, sizeof(uint16_t));
        bytes_read += file.gcount();
        file.read((char *)&chunky, sizeof(uint16_t));
        bytes_read += file.gcount();
        file.read((char *)&chunkz, sizeof(uint16_t));
        bytes_read += file.gcount();
        file.read((char *)&sizex, sizeof(uint64_t));
        bytes_read += file.gcount();
        file.read((char *)&sizey, sizeof(uint64_t));
        bytes_read += file.gcount();
        file.read((char *)&sizez, sizeof(uint64_t));
        bytes_read += file.gcount();

        file.read((char *)&cropstartx, sizeof(uint64_t));
        bytes_read += file.gcount();
        file.read((char *)&cropendx, sizeof(uint64_t));
        bytes_read += file.gcount();
        file.read((char *)&cropstarty, sizeof(uint64_t));
        bytes_read += file.gcount();
        file.read((char *)&cropendy, sizeof(uint64_t));
        bytes_read += file.gcount();
        file.read((char *)&cropstartz, sizeof(uint64_t));
        bytes_read += file.gcount();
        file.read((char *)&cropendz, sizeof(uint64_t));
        bytes_read += file.gcount();

        // tellg() is -1 after a short read; header_size keeps its last good value
        const std::streamoff read_end = file.tellg();
        file.close();

        if (read_end != (std::streamoff)header_size_expected || bytes_read != header_size_expected)
        {
            std::cerr << "Metadata read failed (short read)" << std::endl;
            // The fields above may now be half new and half old
            mark_unusable("short read");
            return;
        }

        if (chunkx == 0 || chunky == 0 || chunkz == 0)
        {
            mark_unusable("chunk size 0");
            return;
        }

        header_size = read_end;

        countx = (sizex + ((size_t)chunkx) - 1) / ((size_t)chunkx);
        county = (sizey + ((size_t)chunky) - 1) / ((size_t)chunky);
        countz = (sizez + ((size_t)chunkz) - 1) / ((size_t)chunkz);

        max_chunk_size = channel_count * chunkx * chunky * chunkz * sizeof(uint16_t);

        is_valid = true;

        if (reset_cache)
        {
            clear_cache_lines();
        }
    }

    // Nothing may index this mchunk until a later reload succeeds: reads of
    // it return zeros and writes to it are refused. A reader built this way
    // is dropped by get_mchunk, as one whose .meta cannot be opened is.
    void mark_unusable(const char *reason)
    {
        is_valid = false;

        std::string note;
        if (log_limit_header.allow(note))
        {
            std::cerr << "Mchunk header unusable (" << reason << "): " << meta_fname << note << std::endl;
        }
    }

    // Re-reads the header if the .meta changed on disk since it was last read
    void reload_if_modified()
    {
        if (check_mtime_hasmodified())
        {
            std::cerr << "Metadata file modified on disk for " << meta_fname << ", reloading metadata" << std::endl;
            reload_metadata();
        }
    }

    size_t find_index(size_t x, size_t y, size_t z)
    {
        size_t ix = x / chunkx;
        size_t iy = y / chunky;
        size_t iz = z / chunkz;

        return (ix * countz * county) + (iy * countz) + iz;
    }

    // *failed is set when the entry could not be read, as opposed to a chunk
    // that was never written (size 0).
    metadata_entry *load_meta_entry(size_t id, bool *failed = nullptr)
    {
        metadata_entry *out = (metadata_entry *)malloc(sizeof(metadata_entry));
        if (out == NULL)
        {
            if (failed != nullptr)
                *failed = true;
            return NULL;
        }
        out->offset = 0;
        out->size = 0;

        const size_t offset = header_size + (entry_file_line_size * id);

        bool read_ok = false;
        for (size_t i = 0; i < IO_RETRY_COUNT; i++)
        {
            std::ifstream file(meta_fname, std::ios::in | std::ios::binary);

            if (file.fail())
            {
                std::cerr << "Fopen failed (metadata)" << std::endl;
                continue;
            }

            reload_if_modified();

            if (!is_valid)
            {
                // The header could not be read, so where the entry is is unknown
                break;
            }

            file.seekg(offset);
            file.read((char *)&(out->offset), sizeof(uint64_t));
            std::streamsize bytes_read = file.gcount();
            file.read((char *)&(out->size), sizeof(uint32_t));
            bytes_read += file.gcount();
            file.close();

            if (bytes_read != sizeof(uint32_t) + sizeof(uint64_t))
            {
                std::cerr << "Metadata read failed (short read)" << std::endl;
                continue;
            }

            read_ok = true;
            break;
        }

        if (!read_ok)
        {
            // A short read can leave part of a size behind
            out->offset = 0;
            out->size = 0;
            if (failed != nullptr)
                *failed = true;
        }

        return out;
    }

    bool replace_meta_entry(size_t id, metadata_entry *new_entry)
    {
        if (!is_valid)
        {
            // The header could not be read, so where the entry is is unknown
            return false;
        }

        const size_t offset = header_size + (entry_file_line_size * id);

        for (size_t i = 0; i < IO_RETRY_COUNT; i++)
        {
            std::fstream file(meta_fname, std::ios::in | std::ios::out | std::ios::binary);

            if (file.fail())
            {
                std::cerr << "Fopen failed (metadata write)" << std::endl;
                continue;
            }

            file.seekp(offset);
            file.write((char *)&(new_entry->offset), sizeof(uint64_t));
            file.write((char *)&(new_entry->size), sizeof(uint32_t));
            file.close();

            if (file.fail())
            {
                std::string note;
                if (log_limit_entry_write.allow(note))
                {
                    std::cerr << "Metadata entry write failed: " << meta_fname << " chunk " << id << note << std::endl;
                }
                return false;
            }
            return true;
        }

        return false;
    }

    std::mutex chunk_cache_mutex;
    std::deque<std::tuple<size_t, uint16_t *>> chunk_cache;

    // Returns the chunk, or zeros when it was never written or cannot be read
    // or decoded; *failed tells those two apart. NULL only if even the zero
    // buffer cannot be allocated.
    uint16_t *load_chunk(size_t id, size_t sizex, size_t sizey, size_t sizez, bool *failed = nullptr)
    {
        const size_t out_buffer_size = sizex * sizey * sizez * sizeof(uint16_t);
        uint16_t *out = (uint16_t *)calloc(out_buffer_size, 1);
        if (out == NULL)
        {
            std::string note;
            if (log_limit_chunk_oom.allow(note))
            {
                std::cerr << "Chunk read failed (out of memory): " << data_fname << " chunk " << id << note << std::endl;
            }
            if (failed != nullptr)
                *failed = true;
            return NULL;
        }

        // Before the entry is read; see data_gen
        const uint64_t gen = data_gen.load();

        bool entry_failed = false;
        metadata_entry *sel = load_meta_entry(id, &entry_failed);

        if (sel == NULL || sel->size == 0)
        {
            // Never written, or the entry could not be read
            if (entry_failed && failed != nullptr)
                *failed = true;
            free(sel);
            return out;
        }

        uint16_t *from_cache = 0;

        {
            std::unique_lock<std::timed_mutex> lock(global_chunk_cache_mutex, cache_lock_timeout);
            if (lock.owns_lock())
            {
                const global_chunk_line *line = global_cache_find(this_mchunk_id, id);
                if (line != nullptr && line->sizex == sizex && line->sizey == sizey && line->sizez == sizez)
                {
                    from_cache = line->ptr;
                    memcpy((void *)out, (void *)from_cache, out_buffer_size);
                }
            }
        }

        // Either from_cache has the chunk, or was not in cache, or failed to get lock

        if (from_cache == 0)
        {
            // Read from file
            size_t buffer_size = sel->size;
            uint16_t *read_buffer = (uint16_t *)malloc(buffer_size);
            if (read_buffer == NULL)
            {
                std::string note;
                if (log_limit_chunk_oom.allow(note))
                {
                    std::cerr << "Chunk read failed (out of memory): " << data_fname << " chunk " << id << note << std::endl;
                }
                if (failed != nullptr)
                    *failed = true;
                free(sel);
                return out;
            }

            bool read_failed = true;
            for (size_t i = 0; i < IO_RETRY_COUNT; i++)
            {
                std::ifstream file(data_fname, std::ios::in | std::ios::binary);

                if (file.fail())
                {
                    // std::cerr << "Fopen failed" << std::endl;
                    continue;
                }

                file.seekg(sel->offset);
                file.read((char *)read_buffer, sel->size);
                std::streamsize bytes_read = file.gcount();
                file.close();
                if (bytes_read != sel->size)
                {
                    // std::cerr << "Read failed (short read)" << std::endl;
                    continue;
                }
                read_failed = false;
                break;
            }

            // Check for read failure
            if (read_failed)
            {
                // std::cerr << "Read failed (max retries)" << std::endl;
                if (failed != nullptr)
                    *failed = true;
                free(read_buffer);
                free(sel);
                return out;
            }

            // Decompress
            size_t decomp_size = 0;
            char *read_decomp_buffer = NULL;
            pixtype *read_decomp_buffer_pt;
            const char *decode_error = NULL;

            uint32_t height, width, depth = 0;

            // 1 -> zstd
            // 2 -> 264
            // 3 -> AV1
            switch (compression_type)
            {
            case 1:
                // Decompress with ZSTD
                read_decomp_buffer = (char *)calloc(out_buffer_size, 1);
                if (read_decomp_buffer == NULL)
                {
                    decode_error = "out of memory";
                    break;
                }
                decomp_size = ZSTD_decompress(read_decomp_buffer, out_buffer_size, read_buffer, sel->size);
                if (ZSTD_isError(decomp_size))
                {
                    decode_error = ZSTD_getErrorName(decomp_size);
                }
                break;

            case 2:
            case 3:
            {
                // decode_stack_native reads a fixed header (13 uint32 fields and
                // a uint64 size) without checking the buffer's length
                if (sel->size < 13 * sizeof(uint32_t) + sizeof(uint64_t))
                {
                    decode_error = "video frame shorter than its header";
                    break;
                }

                // Decompress with vidlib 2
                // read_decomp_buffer_pt = decode_stack_AV1(sizex, sizey, sizez, read_buffer, sel->size);
                auto decode_result = decode_stack_native(read_buffer, sel->size);

                read_decomp_buffer_pt = std::get<0>(decode_result);
                decomp_size = std::get<1>(decode_result);

                width = std::get<0>(std::get<2>(decode_result));
                height = std::get<1>(std::get<2>(decode_result));
                depth = std::get<2>(std::get<2>(decode_result));

                if (read_decomp_buffer_pt == NULL)
                {
                    decode_error = "video decode returned no frames";
                }
                else if (std::get<3>(decode_result) == sizeof(uint8_t))
                {
                    read_decomp_buffer = (char *)uint8_to_uint16_crop(read_decomp_buffer_pt, decomp_size, width, height, depth, sizex, sizey, sizez);
                    decomp_size = sizex * sizey * sizez * sizeof(uint16_t);
                    free(read_decomp_buffer_pt);
                }
                else if (std::get<3>(decode_result) == sizeof(uint16_t))
                {
                    read_decomp_buffer = (char *)uint16_to_uint16_crop((uint16_t *)read_decomp_buffer_pt, decomp_size, width, height, depth, sizex, sizey, sizez);
                    decomp_size = sizex * sizey * sizez * sizeof(uint16_t);
                    free(read_decomp_buffer_pt);
                }
                else
                {
                    std::cerr << "decode_stack_native returned unexpected pixel size" << std::endl;
                    decode_error = "unexpected pixel size";
                    free(read_decomp_buffer_pt);
                }

                break;
            }

            default:
                decode_error = "unknown compression type";
                break;
            }

            free(read_buffer);

            if (decode_error == NULL && decomp_size != out_buffer_size)
            {
                decode_error = "decoded size does not match the chunk";
            }

            if (decode_error != NULL)
            {
                std::string note;
                if (log_limit_chunk_decode.allow(note))
                {
                    std::cerr << "Chunk decode failed (" << decode_error << "): " << data_fname << " chunk " << id
                              << " compression " << compression_type << " decoded " << decomp_size << " expected " << out_buffer_size
                              << note << std::endl;
                }
                if (failed != nullptr)
                    *failed = true;
                free(read_decomp_buffer);
                free(sel);
                return out;
            }

            // Copy result
            memcpy((void *)out, (void *)read_decomp_buffer, decomp_size);

            {
                std::unique_lock<std::timed_mutex> lock(global_chunk_cache_mutex, cache_lock_timeout);
                // If data_gen moved, a write to this mchunk finished while this chunk
                // was read, so this decode may be the chunk it replaced
                if (lock.owns_lock() && data_gen.load() == gen)
                {
                    uint16_t *line_ptr = (uint16_t *)read_decomp_buffer;
                    read_decomp_buffer = NULL;
                    global_cache_insert(this_mchunk_id, id, line_ptr, sizex, sizey, sizez);
                }
            }
            free(read_decomp_buffer);
        }

        free(sel);

        return out;
    }

    // Returns false if the chunk may not be on disk as requested
    bool overwrite_chunk(size_t id, uint16_t *data, size_t data_size)
    {
        // compress data using ZSTD
        size_t compressed_size = ZSTD_compressBound(data_size);
        void *compressed_data = malloc(compressed_size);
        if (compressed_data == NULL)
        {
            std::string note;
            if (log_limit_write_oom.allow(note))
            {
                std::cerr << "Chunk write failed (out of memory): " << data_fname << " chunk " << id << note << std::endl;
            }
            return false;
        }

        compressed_size = ZSTD_compress(compressed_data, compressed_size, data, data_size, 5);
        if (ZSTD_isError(compressed_size))
        {
            std::cerr << "ZSTD_compress failed" << std::endl;
            free(compressed_data);
            return false;
        }

        bool entry_written;
        {
            // Released on every return and if anything below throws
            std::lock_guard<std::timed_mutex> lock(global_chunk_cache_mutex);

            // Write to file
            std::fstream file(data_fname, std::ios::in | std::ios::out | std::ios::binary);
            if (file.fail())
            {
                std::cerr << "Fopen failed (write)" << std::endl;
                free(compressed_data);
                return false;
            }

            file.seekp(0, std::ios::end);
            size_t new_offset = file.tellp();
            // file.seekp(sel->offset);
            file.write((char *)compressed_data, compressed_size);
            file.close();

            // Never point the entry at bytes that did not reach the file
            if (new_offset == (size_t)-1 || file.fail())
            {
                std::string note;
                if (log_limit_append.allow(note))
                {
                    std::cerr << "Chunk append failed: " << data_fname << " chunk " << id << note << std::endl;
                }
                free(compressed_data);
                return false;
            }

            metadata_entry new_entry;
            new_entry.offset = new_offset;
            new_entry.size = compressed_size;

            entry_written = replace_meta_entry(id, &new_entry);

            // Update cached mtime so our own write is not detected as
            // an external modification on the next read.
            last_meta_read_time = get_file_mtime_ns(meta_fname);

            // After the entry is on disk and before the lines are dropped: a
            // reader that sampled data_gen before this will not insert
            data_gen.fetch_add(1);

            // Delete the prexisting values in the cache
            global_chunk_line *line = global_cache_find(this_mchunk_id, id);
            if (line != nullptr)
            {
                global_cache_drop(line - global_chunk_cache);
            }
        }

        free(compressed_data);
        return entry_written;
    }

    uint16_t read_pixel(size_t i, size_t j, size_t k)
    {
        const size_t xmin = ((size_t)chunkx) * (i / ((size_t)chunkx));
        const size_t xmax = std::min((size_t)xmin + chunkx, (size_t)sizex);
        const size_t xsize = xmax - xmin;

        const size_t ymin = ((size_t)chunky) * (j / ((size_t)chunky));
        const size_t ymax = std::min((size_t)ymin + chunky, (size_t)sizey);
        const size_t ysize = ymax - ymin;

        const size_t zmin = ((size_t)chunkz) * (k / ((size_t)chunkz));
        const size_t zmax = std::min((size_t)zmin + chunkz, (size_t)sizez);
        const size_t zsize = zmax - zmin;

        //                   X                              Y                    Z
        const size_t coffset = ((i - xmin) * ysize * zsize) + ((j - ymin) * zsize) + (k - zmin);

        const size_t chunk_id = find_index(i, j, k);

        uint16_t *chunk = load_chunk(chunk_id, xsize, ysize, zsize);

        const uint16_t out = chunk[coffset];

        free(chunk);

        return out;
    }

    void print_info()
    {
        /*
        std::cout << "-----------------------------------" << std::endl;
        std::cout << "Files: " << meta_fname << ", " << data_fname << std::endl;
        std::cout << "dtype = " << dtype << std::endl;
        std::cout << "Chunks: " << chunkx << ", " << chunky << ", " << chunkz << std::endl;
        std::cout << "Size: " << sizex << ", " << sizey << ", " << sizez << std::endl;
        std::cout << "Tile Counts: " << countx << ", " << county << ", " << countz << std::endl;
        std::cout << "-----------------------------------" << std::endl;
        */

        std::cout << '[' << data_fname << "] " << " dtype=" << dtype << " chunks=(" << chunkx << ','
                  << chunky << ',' << chunkz << ") size=(" << sizex << ',' << sizey << ',' << sizez << ") "
                  << "tile_counts=(" << countx << ',' << county << ',' << countz << ")" << std::endl;
    }
};

class descriptor_layer
{
public:
    std::string source_name;
    uint16_t source_channel = 0;
    uint16_t target_channel = 0;

    size_t sizex = 0;
    size_t sizey = 0;
    size_t sizez = 0;

    // TODO, not currently used
    int64_t ioffsetx = 0;
    int64_t ioffsety = 0;
    int64_t ioffsetz = 0;

    int64_t ooffsetx = 0;
    int64_t ooffsety = 0;
    int64_t ooffsetz = 0;

    bool invertx = false;
    bool inverty = false;
    bool invertz = false;

    descriptor_layer()
    {
    }
};

// A chunk decoded for one request, with the extent it was decoded with. A
// reload in the middle of a request can change the extent of the chunk a key
// names, and a buffer must only ever be indexed with its own extent.
struct region_chunk
{
    uint16_t *ptr = nullptr;
    size_t xmin = 0, ymin = 0, zmin = 0;
    size_t xsize = 0, ysize = 0, zsize = 0;

    bool has_extent(size_t x0, size_t y0, size_t z0, size_t sx, size_t sy, size_t sz) const
    {
        return xmin == x0 && ymin == y0 && zmin == z0 && xsize == sx && ysize == sy && zsize == sz;
    }
};

class archive_reader
{
public:
    bool is_protected = false;
    bool is_valid = true;
    bool metadata_json = false;

    std::string fname; // "./example_dset"
    uint16_t channel_count;
    uint16_t archive_version; // 1 == current
    uint16_t dtype;
    uint16_t mchunkx, mchunky, mchunkz;
    uint64_t resx, resy, resz;
    uint64_t sizex, sizey, sizez;
    uint64_t mcountx, mcounty, mcountz;

    std::vector<size_t> scales;

    std::vector<descriptor_layer *> descriptor_layers;
    std::map<uint16_t, uint16_t> descriptor_channel_map;
    int64_t descriptor_x_origin;
    int64_t descriptor_y_origin;
    int64_t descriptor_z_origin;

    ArchiveType type;

    // The archive geometry is read once per process. Its file and mtime at
    // that point let a read outside the stored tiles say whether it is stale.
    std::string metadata_fname;
    int64_t metadata_mtime_ns = 0;

    std::unordered_map<std::string, archive_reader *> *parent_archive_inventory;

    archive_reader(std::string name_in, enum ArchiveType type_in, std::unordered_map<std::string, archive_reader *> *par_in = nullptr)
    {
        fname = name_in;
        type = type_in;
        parent_archive_inventory = par_in;

        metadata_json = false;
        if (type == SISF_JSON)
        {
            // SISF_JSON is parsed the same other than metadata, set flag and type
            metadata_json = true;
            type = SISF;
        }

        switch (type)
        {
        case SISF:
            load_metadata_sisf();
            break;
        case ZARR:
            load_metadata_zarr();
            break;
        case DESCRIPTOR:
            load_metadata_descriptor();
            break;
        }

        load_protection();
    }

    ~archive_reader() {}

    // Return true if the contents of filters allows access to this dataset
    bool verify_protection(std::vector<std::pair<std::string, std::string>> filters)
    {
        if (!this->is_protected)
        {
            return true;
        }

        std::string token_in = "";
        for (auto i : filters)
        {
            if (i.first == "token")
            {
                token_in = i.second;
            }
        }

        if (token_in.size() == 0)
        {
            return false;
        }

        std::ifstream access_file(fname + "/.sisf_access");

        std::string line;
        while (std::getline(access_file, line))
        {
            if (line.size() == 0)
            {
                continue;
            }

            if (line == token_in)
                return true;
        }

        return false;
    }

    void load_protection()
    {
        std::vector<std::string> fnames = glob_tool(std::string(fname + "/.sisf_access"));

        if (fnames.size() > 0)
        {
            this->is_protected = true;
        }
        else
        {
            this->is_protected = false;
        }
    }

    void load_metadata_zarr()
    {
        tensorstore::Context context = tensorstore::Context::Default();
        auto store_future = tensorstore::Open({{"driver", "zarr3"},
                                               {"kvstore", {{"driver", "file"}, {"path", fname}}}},
                                              context, tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read);

        auto store_result = store_future.result();

        if (!store_result.ok())
        {
            std::cerr << "Error opening TensorStore: " << store_result.status() << std::endl;
        }

        auto store = std::move(store_result.value());

        auto domain = store.domain();
        auto shape = domain.shape();
        auto labels = domain.labels();

        sizex = 0;
        sizey = 0;
        sizez = 0;
        channel_count = 0;

        size_t i = 0;
        for (const auto &dim : shape)
        {
            switch (i)
            {
            case 0: // x
                sizex = dim;
                break;
            case 1: // y
                sizey = dim;
                break;
            case 2: // z
                sizez = dim;
                break;
            case 3: // c
                channel_count = dim;
                break;
            }

            i++;
        }

        if (sizex == 0 || sizey == 0 || sizez == 0 || channel_count == 0)
        {
            std::cerr << "Invalid rank in Zarr dataset [" << fname << "]. Found i=" << i << std::endl;
        }

        archive_version = 0;
        dtype = 1;

        // TODO Load res from ts
        resx = 100;
        resy = 100;
        resz = 100;

        auto dim_units_result = store.dimension_units();

        if (!dim_units_result.ok())
        {
            std::cerr << "Error reading dimension_units from TensorStore: " << store_result.status() << std::endl;
        }
        else
        {
            auto dim_units = dim_units_result.value();

            for (size_t i = 0; i < dim_units.size(); i++)
            {
                if (dim_units[i].has_value())
                {
                    tensorstore::Unit u = dim_units[i].value();

                    // TODO Verify that unit is nm and scale properly

                    switch (i)
                    {
                    case 0: // x
                        resx = u.multiplier;
                        break;
                    case 1: // y
                        resy = u.multiplier;
                        break;
                    case 2: // z
                        resz = u.multiplier;
                        break;
                    case 3: // c
                        // Color does not have a unit
                        break;
                    }

                    // std::cout << "Name: " << labels[i] << '\t' << u.to_string() << '\t' << u.base_unit << '\t' << u.multiplier << std::endl;
                }
            }
        }

        mchunkx = sizex;
        mchunky = sizey;
        mchunkz = sizez;

        mcountx = 1;
        mcounty = 1;
        mcountz = 1;

        scales.push_back(1);
    }

    // The dataset cannot be served. load_inventory catches this and prints
    // [FAIL] for it, so it is not added, requests for it answer 404, and the
    // next inventory scan tries it again.
    [[noreturn]] void reject_metadata(const char *reason)
    {
        std::string note;
        if (log_limit_archive_meta.allow(note))
        {
            std::cerr << "Dataset metadata unusable (" << reason << "): " << metadata_fname << note << std::endl;
        }
        throw std::runtime_error("unusable dataset metadata");
    }

    void load_metadata_sisf()
    {
        metadata_fname = fname + (metadata_json ? "/metadata.json" : "/metadata.bin");
        metadata_mtime_ns = get_file_mtime_ns(metadata_fname);

        if (metadata_json)
        {
            std::ifstream inputFile(fname + "/metadata.json");
            if (!inputFile)
            {
                ; // TODO error handling
            }

            json jsonData;
            inputFile >> jsonData; // Read JSON data from file

            archive_version = 2;
            dtype = 1;
            channel_count = jsonData["channel_count"];

            mchunkx = jsonData["xchunk"];
            mchunky = jsonData["ychunk"];
            mchunkz = jsonData["zchunk"];

            sizex = jsonData["xsize"];
            sizey = jsonData["ysize"];
            sizez = jsonData["zsize"];

            resx = jsonData["xres"];
            resy = jsonData["yres"];
            resz = jsonData["zres"];

            // std::cout << "Read JSON from file: " << jsonData.dump(4) << std::endl;
        }
        else
        {
            std::ifstream file(fname + "/metadata.bin", std::ios::in | std::ios::binary);

            if (file.fail())
            {
                ;
            }

            std::streamsize bytes_read = 0;
            file.read((char *)&archive_version, sizeof(uint16_t));
            bytes_read += file.gcount();
            file.read((char *)&dtype, sizeof(uint16_t));
            bytes_read += file.gcount();
            file.read((char *)&channel_count, sizeof(uint16_t));
            bytes_read += file.gcount();

            file.read((char *)&mchunkx, sizeof(uint16_t));
            bytes_read += file.gcount();
            file.read((char *)&mchunky, sizeof(uint16_t));
            bytes_read += file.gcount();
            file.read((char *)&mchunkz, sizeof(uint16_t));
            bytes_read += file.gcount();
            file.read((char *)&resx, sizeof(uint64_t));
            bytes_read += file.gcount();
            file.read((char *)&resy, sizeof(uint64_t));
            bytes_read += file.gcount();
            file.read((char *)&resz, sizeof(uint64_t));
            bytes_read += file.gcount();
            file.read((char *)&sizex, sizeof(uint64_t));
            bytes_read += file.gcount();
            file.read((char *)&sizey, sizeof(uint64_t));
            bytes_read += file.gcount();
            file.read((char *)&sizez, sizeof(uint64_t));
            bytes_read += file.gcount();

            // Unopenable, empty (e.g. mid-write) or cut short: the fields
            // above were not all read
            if (bytes_read != sizeof(uint16_t) * 6 + sizeof(uint64_t) * 6)
            {
                reject_metadata("short read");
            }
        }

        // The mchunk counts below divide by these
        if (mchunkx == 0 || mchunky == 0 || mchunkz == 0)
        {
            reject_metadata("mchunk size 0");
        }

        mcountx = (sizex + mchunkx - 1) / mchunkx;
        mcounty = (sizey + mchunky - 1) / mchunky;
        mcountz = (sizez + mchunkz - 1) / mchunkz;

        // Find resolution tiers
        std::vector<std::string> fnames = glob_tool(std::string(fname + "/meta/*.meta"));

        for (std::vector<std::string>::iterator i = fnames.begin(); i != fnames.end(); i++)
        {
            size_t loc1 = i->find_last_of("/");
            size_t loc2 = i->find_last_of('.');

            const size_t label_offset = 6; // "chunk_"

            std::string ii = std::string(i->c_str() + loc1 + label_offset + 1, i->c_str() + loc2);

            std::string scale_label = ii.substr(ii.find_last_of('.') + 1);
            const size_t scale = stoi(scale_label);
            const size_t cnt = std::count(scales.begin(), scales.end(), scale);

            if (cnt == 0)
            {
                scales.push_back(scale);
            }
        }

        std::sort(scales.begin(), scales.end());
    }

    void load_metadata_descriptor()
    {
        std::ifstream inputFile(fname + "/descriptor.json");
        if (!inputFile)
        {
            ; // TODO error handling
        }

        json jsonData;
        inputFile >> jsonData; // Read JSON data from file

        scales.push_back(1);

        mcountx = 0;
        mcounty = 0;
        mcountz = 0;

        mchunkx = 0;
        mchunky = 0;
        mchunkz = 0;

        archive_version = 2;
        dtype = 1;

        resx = jsonData["xres"];
        resy = jsonData["yres"];
        resz = jsonData["zres"];

        json layers = jsonData["layers"];

        channel_count = 0;
        if (layers.is_array())
        {
            for (const auto &element : layers)
            {
                descriptor_layer *layer = new descriptor_layer();

                layer->source_name = element["source"];
                layer->source_channel = element["source_channel"];
                layer->target_channel = element["target_channel"];

                json source_size = element["source_size"];
                if (source_size.is_array())
                {
                    size_t i = 0;

                    for (const auto &a : source_size)
                    {
                        int s = a.get<int>();

                        switch (i)
                        {
                        case 0:
                            layer->sizex = s;
                            break;
                        case 1:
                            layer->sizey = s;
                            break;
                        case 2:
                            layer->sizez = s;
                            break;
                        default:
                            break;
                        }

                        i++;
                    }

                    if (i != 3)
                    {
                        throw std::runtime_error("invalid source_size size");
                    }
                }
                else
                {
                    throw std::runtime_error("source_size should be an array");
                }

                json out_size = element["target_offset"];
                if (out_size.is_array())
                {
                    size_t i = 0;

                    for (const auto &a : out_size)
                    {
                        int s = a.get<int>();

                        switch (i)
                        {
                        case 0:
                            layer->ooffsetx = s;
                            break;
                        case 1:
                            layer->ooffsety = s;
                            break;
                        case 2:
                            layer->ooffsetz = s;
                            break;
                        default:
                            break;
                        }

                        i++;
                    }

                    if (i != 3)
                    {
                        throw std::runtime_error("invalid out_size size");
                    }
                }
                else
                {
                    throw std::runtime_error("out_size should be an array");
                }

                descriptor_layers.push_back(layer);
            }
        }
        else
        {
            throw std::runtime_error("layer is not an array");
        }

        int64_t minx = 0; // std::numeric_limits<int64_t>::max();
        int64_t maxx = std::numeric_limits<int64_t>::min();

        int64_t miny = 0; // std::numeric_limits<int64_t>::max();
        int64_t maxy = std::numeric_limits<int64_t>::min();

        int64_t minz = 0; // std::numeric_limits<int64_t>::max();
        int64_t maxz = std::numeric_limits<int64_t>::min();

        std::set<uint16_t> found_channels;
        for (descriptor_layer *l : descriptor_layers)
        {
            uint16_t tc = l->target_channel;
            found_channels.insert(tc);
            if (descriptor_channel_map.count(tc) == 0)
            {
                descriptor_channel_map[tc] = found_channels.size() - 1;
            }

            minx = std::min(minx, l->ooffsetx);
            miny = std::min(miny, l->ooffsety);
            minz = std::min(minz, l->ooffsetz);

            maxx = std::max(maxx, static_cast<int64_t>(l->sizex) + l->ooffsetx);
            maxy = std::max(maxy, static_cast<int64_t>(l->sizey) + l->ooffsety);
            maxz = std::max(maxz, static_cast<int64_t>(l->sizez) + l->ooffsetz);
        }

        sizex = maxx - minx;
        sizey = maxy - miny;
        sizez = maxz - minz;

        descriptor_x_origin = minx;
        descriptor_y_origin = miny;
        descriptor_z_origin = minz;

        channel_count = found_channels.size();
    }

    std::tuple<size_t, size_t, size_t> get_size(size_t scale)
    {
        size_t size_x_out = sizex / scale;
        size_t size_y_out = sizey / scale;
        size_t size_z_out = sizez / scale;

        // Start with the size of one chunk
        double dilation_x = this->mchunkx;
        double dilation_y = this->mchunky;
        double dilation_z = this->mchunkz;

        // Divide by scale
        dilation_x /= scale;
        dilation_y /= scale;
        dilation_z /= scale;

        // Isolate fractional part of dilation
        dilation_x -= std::floor(dilation_x);
        dilation_y -= std::floor(dilation_y);
        dilation_z -= std::floor(dilation_z);

        // Scale dilation by number of tiles
        dilation_x *= this->mcountx;
        dilation_y *= this->mcounty;
        dilation_z *= this->mcountz;

        // round up
        dilation_x = std::ceil(dilation_x);
        dilation_y = std::ceil(dilation_y);
        dilation_z = std::ceil(dilation_z);

        auto sub_clamp = [](size_t a, double b) -> size_t {
            return (size_t)std::max<int64_t>((int64_t)a - (int64_t)b, 1);
        };
        size_x_out = sub_clamp(size_x_out, dilation_x);
        size_y_out = sub_clamp(size_y_out, dilation_y);
        size_z_out = sub_clamp(size_z_out, dilation_z);

        size_x_out = std::max(size_x_out, (size_t)1);
        size_y_out = std::max(size_y_out, (size_t)1);
        size_z_out = std::max(size_z_out, (size_t)1);

        // std::cout << "Scale: " << scale << " " << dilation_x << " " << dilation_y << " " << dilation_z << std::endl;

        return std::make_tuple(size_x_out, size_y_out, size_z_out);
    }

    std::tuple<size_t, size_t, size_t> get_res(size_t scale)
    {
        std::tuple<size_t, size_t, size_t> size = this->get_size(scale);

        double resx_out = resx;
        double resy_out = resy;
        double resz_out = resz;

        resx_out *= sizex;
        resy_out *= sizey;
        resz_out *= sizez;

        resx_out /= std::get<0>(size);
        resy_out /= std::get<1>(size);
        resz_out /= std::get<2>(size);

        return std::make_tuple((size_t)resx_out, (size_t)resy_out, (size_t)resz_out);
    }

    bool contains_scale(size_t scale)
    {
        return std::find(scales.begin(), scales.end(), scale) != scales.end();
    }

    std::tuple<size_t, size_t, size_t> inline find_index(size_t x, size_t y, size_t z)
    {
        size_t ix = x / mchunkx;
        size_t iy = y / mchunky;
        size_t iz = z / mchunkz;

        return std::make_tuple(ix, iy, iz);
    }

    size_t inline pixel_size()
    {
        return sizeof(uint16_t);
    }

    std::map<std::tuple<size_t, size_t, size_t, size_t, size_t>, packed_reader *> mchunk_buffer;
    std::mutex mchunk_buffer_mutex;
    packed_reader *get_mchunk(size_t scale, size_t channel, size_t i, size_t j, size_t k)
    {
        std::tuple<size_t, size_t, size_t, size_t, size_t> id_tuple = std::make_tuple(scale, channel, i, j, k);

        // Released on every return and if anything below throws (e.g. bad_alloc)
        std::lock_guard<std::mutex> lock(mchunk_buffer_mutex);

        packed_reader *out = mchunk_buffer[id_tuple];

        if (out == 0 || out == nullptr)
        {
            std::stringstream ss;
            ss << "chunk_" << i << '_' << j << '_' << k << '.' << channel << '.' << scale << 'X';
            const std::string chunk_root = ss.str();

            const std::string chunk_meta_name = fname + "/meta/" + chunk_root + ".meta";
            const std::string chunk_data_name = fname + "/data/" + chunk_root + ".data";

            mchunk_uuid_mutex.lock();
            size_t random_id = mchunk_uuid;
            mchunk_uuid++;
            mchunk_uuid_mutex.unlock();

            out = new packed_reader(random_id, chunk_meta_name, chunk_data_name);

            if (!out->is_valid)
            {
                delete out;
                out = nullptr;
            }

            mchunk_buffer[id_tuple] = out;
        }

        return out;
    }

    // With failed set, a chunk that cannot be read (the .meta or .data cannot
    // be opened or is short, the data does not decode, memory runs out, the
    // mchunk header is unusable, no reader can be built for its mchunk) sets
    // *failed; its voxels read as 0 either way. A chunk never written is not
    // a failure.
    uint16_t *load_region(
        size_t scale,
        size_t xs, size_t xe,
        size_t ys, size_t ye,
        size_t zs, size_t ze,
        bool *failed = nullptr)
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        // Calculate size of output
        const size_t osizex = xe - xs;
        const size_t osizey = ye - ys;
        const size_t osizez = ze - zs;
        const size_t buffer_size = osizex * osizey * osizez * sizeof(uint16_t) * channel_count;

        // Allocate buffer for output
        uint16_t *out_buffer = (uint16_t *)calloc(buffer_size, 1);
        if (out_buffer == NULL)
        {
            return NULL;
        }

        if (type == SISF)
        {
            // Define map for storing already decompressed chunks
            std::map<std::tuple<size_t, size_t, size_t, size_t, size_t>, region_chunk> chunk_cache;
            std::set<std::tuple<size_t, size_t, size_t, size_t, size_t>> back_mchunks;

            // Scaled metachunk size — clamp to >=1
            const size_t mcx = std::max<size_t>(1, mchunkx / scale);
            const size_t mcy = std::max<size_t>(1, mchunky / scale);
            const size_t mcz = std::max<size_t>(1, mchunkz / scale);

            // Variables to store chunk reader and data (shared in loop)
            packed_reader *chunk_reader = nullptr;
            std::tuple<size_t, size_t, size_t, size_t, size_t> *chunk_identifier = nullptr;
            size_t sub_chunk_id;
            uint16_t *chunk;

            // Variables for tracking the last chunks that were used
            size_t last_x, last_y, last_z, last_sub, last_c;
            size_t cxmin, cxmax, cxsize;
            size_t cymin, cymax, cysize;
            size_t czmin, czmax, czsize;

            // Voxels that fall outside their stored tile read as 0
            size_t outside_voxels = 0;
            // Voxels moved outside their chunk by a reload during this request read as 0
            size_t stale_voxels = 0;

            for (size_t c = 0; c < channel_count; c++)
            {
                for (size_t i = xs; i < xe; i++)
                {
                    const size_t xmin = mcx * (i / mcx);                             // lower bound of mchunk
                    const size_t xmax = std::min((size_t)xmin + mcx, (size_t)sizex); // upper bound of mchunk
                    const size_t xsize = xmax - xmin;                                // size of mchunk
                    const size_t chunk_id_x = i / ((size_t)mcx);                     // mchunk x id
                    const size_t x_in_chunk = i - xmin;                              // x displacement inside chunk

                    for (size_t j = ys; j < ye; j++)
                    {
                        const size_t ymin = mcy * (j / mcy);
                        const size_t ymax = std::min((size_t)ymin + mcy, (size_t)sizey);
                        const size_t ysize = ymax - ymin;
                        const size_t chunk_id_y = j / ((size_t)mcy);
                        const size_t y_in_chunk = j - ymin;

                        for (size_t k = zs; k < ze; k++)
                        {
                            const size_t zmin = mcz * (k / mcz);
                            const size_t zmax = std::min((size_t)zmin + mcz, (size_t)sizez);
                            const size_t zsize = zmax - zmin;
                            const size_t chunk_id_z = k / ((size_t)mcz);
                            const size_t z_in_chunk = k - zmin;

                            bool force = false;
                            if (chunk_reader == nullptr ||
                                chunk_identifier == nullptr ||
                                last_x != chunk_id_x ||
                                last_y != chunk_id_y ||
                                last_z != chunk_id_z ||
                                last_c != c)
                            {
                                force = true;

                                bool is_bad = back_mchunks.count({scale, c, chunk_id_x, chunk_id_y, chunk_id_z}) > 0;

                                if (!is_bad)
                                {
                                    chunk_reader = get_mchunk(scale, c, chunk_id_x, chunk_id_y, chunk_id_z);
                                }
                                else
                                {
                                    chunk_reader = nullptr;
                                }

                                if (chunk_reader == nullptr || chunk_reader == 0)
                                {
                                    // No reader could be built: the .meta is missing, cannot be
                                    // opened or has an unusable header
                                    if (failed != nullptr)
                                        *failed = true;
                                    if (!is_bad)
                                    {
                                        back_mchunks.insert({scale, c, chunk_id_x, chunk_id_y, chunk_id_z});
                                    }
                                    continue;
                                }

                                if (!chunk_reader->is_valid)
                                {
                                    // Its last reload failed; retry if the file changed since
                                    chunk_reader->reload_if_modified();
                                }

                                last_x = chunk_id_x;
                                last_y = chunk_id_y;
                                last_z = chunk_id_z;
                            }

                            if (!chunk_reader->is_valid)
                            {
                                // The mchunk's header could not be read on a reload, so its
                                // geometry is unknown: its voxels read as 0
                                if (failed != nullptr)
                                    *failed = true;
                                back_mchunks.insert({scale, c, chunk_id_x, chunk_id_y, chunk_id_z});
                                chunk_reader = nullptr;
                                continue;
                            }

                            // Shift ranges for cropping
                            const size_t x_in_chunk_offset = x_in_chunk + chunk_reader->cropstartx;
                            const size_t y_in_chunk_offset = y_in_chunk + chunk_reader->cropstarty;
                            const size_t z_in_chunk_offset = z_in_chunk + chunk_reader->cropstartz;

                            // Outside the stored tile, e.g. when the archive geometry is stale after
                            // an in-place re-conversion. find_index would name a chunk that is not this one.
                            if (x_in_chunk_offset >= chunk_reader->sizex ||
                                y_in_chunk_offset >= chunk_reader->sizey ||
                                z_in_chunk_offset >= chunk_reader->sizez)
                            {
                                outside_voxels++;
                                if (force && chunk_identifier != nullptr)
                                {
                                    // Make the next voxel start over; the cached chunk belongs to the previous mchunk
                                    delete chunk_identifier;
                                    chunk_identifier = nullptr;
                                }
                                continue;
                            }

                            // Find sub chunk id from coordinates
                            sub_chunk_id = chunk_reader->find_index(x_in_chunk_offset, y_in_chunk_offset, z_in_chunk_offset);

                            // Only perform this step if there has been a change in chunk
                            if (force ||
                                last_sub != sub_chunk_id)
                            {
                                // Replace the chunk id with the new one
                                if (chunk_identifier != nullptr)
                                {
                                    delete chunk_identifier;
                                }
                                chunk_identifier = new std::tuple(c, chunk_id_x, chunk_id_y, chunk_id_z, sub_chunk_id);

                                // Find the start/stop coordinates of this chunk
                                cxmin = ((size_t)chunk_reader->chunkx) * (x_in_chunk_offset / ((size_t)chunk_reader->chunkx)); // Minimum value of the chunk
                                cxmax = std::min((size_t)cxmin + chunk_reader->chunkx, (size_t)chunk_reader->sizex);           // Maximum value of the chunk
                                cxsize = cxmax - cxmin;                                                                        // Size of the chunk

                                cymin = ((size_t)chunk_reader->chunky) * (y_in_chunk_offset / ((size_t)chunk_reader->chunky));
                                cymax = std::min((size_t)cymin + chunk_reader->chunky, (size_t)chunk_reader->sizey);
                                cysize = cymax - cymin;

                                czmin = ((size_t)chunk_reader->chunkz) * (z_in_chunk_offset / ((size_t)chunk_reader->chunkz));
                                czmax = std::min((size_t)czmin + chunk_reader->chunkz, (size_t)chunk_reader->sizez);
                                czsize = czmax - czmin;

                                if (cxmax <= cxmin || cymax <= cymin || czmax <= czmin)
                                {
                                    outside_voxels++;
                                    delete chunk_identifier;
                                    chunk_identifier = nullptr;
                                    continue;
                                }

                                // Check if the chunk is in the tmp cache
                                region_chunk &cached = chunk_cache[*chunk_identifier];
                                if (cached.ptr != nullptr && !cached.has_extent(cxmin, cymin, czmin, cxsize, cysize, czsize))
                                {
                                    // Decoded before a reload in this request changed its extent
                                    free(cached.ptr);
                                    cached.ptr = nullptr;
                                }
                                if (cached.ptr == nullptr)
                                {
                                    cached = region_chunk{chunk_reader->load_chunk(sub_chunk_id, cxsize, cysize, czsize, failed),
                                                          cxmin, cymin, czmin, cxsize, cysize, czsize};
                                }
                                chunk = cached.ptr;

                                // Store this ID as the most recent chunk
                                last_sub = sub_chunk_id;
                                last_c = c;
                            }

                            if (chunk == nullptr)
                            {
                                // Out of memory in load_chunk; read as 0
                                continue;
                            }

                            // The chunk was sized before a reload moved this voxel outside it
                            if (x_in_chunk_offset - cxmin >= cxsize ||
                                y_in_chunk_offset - cymin >= cysize ||
                                z_in_chunk_offset - czmin >= czsize)
                            {
                                stale_voxels++;
                                if (failed != nullptr)
                                    *failed = true;
                                continue;
                            }

                            // Calculate the coordinates of the input and output inside their respective buffers
                            const size_t coffset = ((x_in_chunk_offset - cxmin) * cysize * czsize) + // X
                                                   ((y_in_chunk_offset - cymin) * czsize) +          // Y
                                                   (z_in_chunk_offset - czmin);                      // Z

                            const size_t ooffset = (c * osizey * osizex * osizez) + // C
                                                   ((k - zs) * osizey * osizex) +   // Z
                                                   ((j - ys) * osizex) +            // Y
                                                   ((i - xs));                      // X

                            out_buffer[ooffset] = chunk[coffset];
                        }
                    }
                }
            }

            if (chunk_identifier != nullptr)
            {
                delete chunk_identifier;
            }

            for (auto it = chunk_cache.begin(); it != chunk_cache.end(); it++)
            {
                free(it->second.ptr);
            }

            std::string note;
            if (outside_voxels > 0)
            {
                if (log_limit_read_outside.allow(note))
                {
                    std::cerr << "Read outside stored tiles: " << outside_voxels << " voxels read as 0 in " << fname
                              << " scale " << scale << " box " << xs << '-' << xe << '_' << ys << '-' << ye << '_' << zs << '-' << ze
                              << " (archive geometry may be stale)" << note << std::endl;
                }

                // This process keeps the tile step and size it read at startup; only a restart reads them again
                if (get_file_mtime_ns(metadata_fname) != metadata_mtime_ns && log_limit_stale_archive.allow(note))
                {
                    std::cerr << "Archive geometry is stale: " << metadata_fname << " changed on disk after it was loaded; "
                              << "restart the CDN to serve the new geometry of " << fname << note << std::endl;
                }
            }

            if (stale_voxels > 0 && log_limit_stale_extent.allow(note))
            {
                std::cerr << "Chunk extent changed during a read: " << stale_voxels << " voxels read as 0 in " << fname
                          << " scale " << scale << " box " << xs << '-' << xe << '_' << ys << '-' << ye << '_' << zs << '-' << ze
                          << note << std::endl;
            }

            if (failed != nullptr && *failed && log_limit_read_refused.allow(note))
            {
                std::cerr << "Read failed (could not read chunk): " << fname
                          << " scale " << scale << " box " << xs << '-' << xe << '_' << ys << '-' << ye << '_' << zs << '-' << ze
                          << note << std::endl;
            }
        }
        else if (type == ZARR)
        {
            tensorstore::Context context = tensorstore::Context::Default();
            auto store_future = tensorstore::Open({{"driver", "zarr3"},
                                                   {"kvstore", {{"driver", "file"}, {"path", fname}}}},
                                                  context, tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read);

            auto store_result = store_future.result();

            if (!store_result.ok())
            {
                std::cerr << "Error opening TensorStore: " << store_result.status() << std::endl;
            }
            else
            {
                auto store = std::move(store_result.value());

                const size_t read_buffer_size = osizex * osizey * osizez * sizeof(uint16_t) * channel_count;
                uint16_t *read_buffer = (uint16_t *)malloc(read_buffer_size);

                auto array_result = tensorstore::Read<tensorstore::zero_origin>(
                                        store | tensorstore::AllDims().SizedInterval(
                                                    {(tensorstore::Index)xs, (tensorstore::Index)ys, (tensorstore::Index)zs, 0},
                                                    {(tensorstore::Index)osizex, (tensorstore::Index)osizey, (tensorstore::Index)osizez, (tensorstore::Index)channel_count}))
                                        .result();

                if (array_result.ok())
                {
                    // tensorstore::Array<tensorstore::Shared<void>, -1, tensorstore::ArrayOriginKind::offset, tensorstore::ContainerKind::container>
                    auto array = array_result.value();

                    // TODO detect datatype automatically
                    uint16_t *array_ptr = (uint16_t *)array.data();

                    // std::cout << "s:" << array.num_elements() << std::endl;
                    // Access example: std::cout << "T: " << array[{xs, ys, zs, 0}] << std::endl;

                    for (size_t c = 0; c < channel_count; c++)
                    {
                        for (size_t i = xs; i < xe; i++)
                        {
                            for (size_t j = ys; j < ye; j++)
                            {
                                for (size_t k = zs; k < ze; k++)
                                {
                                    // Calculate the coordinates of the input and output inside their respective buffers
                                    const size_t coffset = ((i - xs) * osizey * osizez * channel_count) + // X
                                                           ((j - ys) * osizez * channel_count) +          // Y
                                                           (k - zs) * channel_count + c;                  // Z and C

                                    const size_t ooffset = (c * osizey * osizex * osizez) + // C
                                                           ((k - zs) * osizey * osizex) +   // Z
                                                           ((j - ys) * osizex) +            // Y
                                                           ((i - xs));                      // X

                                    out_buffer[ooffset] = array_ptr[coffset];
                                }
                            }
                        }
                    }
                }
                else
                {
                    std::cerr << "Error reading from TensorStore: " << array_result.status() << std::endl;
                }
            }
        }
        else if (type == DESCRIPTOR && parent_archive_inventory != nullptr)
        {
            for (descriptor_layer *l : descriptor_layers)
            {
                // Find the beginning and end of this layer's output, measured relative to the origin (i.e. should never be less than zero)
                const int64_t layer_start_x = l->ooffsetx - descriptor_x_origin;
                const int64_t layer_end_x = layer_start_x + l->sizex;
                const int64_t layer_start_y = l->ooffsety - descriptor_y_origin;
                const int64_t layer_end_y = layer_start_y + l->sizey;
                const int64_t layer_start_z = l->ooffsetz - descriptor_y_origin;
                const int64_t layer_end_z = layer_start_z + l->sizez;

                // Calculate the overlap start-stops, in output space
                const int64_t x_overlap_start = std::max(layer_start_x, static_cast<int64_t>(xs));
                const int64_t x_overlap_end = std::min(layer_end_x, static_cast<int64_t>(xe));
                const int64_t y_overlap_start = std::max(layer_start_y, static_cast<int64_t>(ys));
                const int64_t y_overlap_end = std::min(layer_end_y, static_cast<int64_t>(ye));
                const int64_t z_overlap_start = std::max(layer_start_z, static_cast<int64_t>(zs));
                const int64_t z_overlap_end = std::min(layer_end_z, static_cast<int64_t>(ze));

                // True if there is an overlap between the requested region and the layer region
                const bool overlaps = (x_overlap_start <= x_overlap_end) &&
                                      (y_overlap_start <= y_overlap_end) &&
                                      (z_overlap_start <= z_overlap_end);

                if (!overlaps)
                {
                    // This layer is not included in the current access
                    continue;
                }

                auto reader = parent_archive_inventory->find(l->source_name);

                if (reader == parent_archive_inventory->end())
                {
                    // Source not in inventory
                    continue;
                }

                const size_t scale = 1; // TODO, not currently implemented

                // Calculate the overlap start-stop, in input space
                const int64_t x_overlap_start_shifted = x_overlap_start + l->ioffsetx;
                const int64_t x_overlap_end_shifted = x_overlap_end + l->ioffsetx;
                const int64_t y_overlap_start_shifted = y_overlap_start + l->ioffsety;
                const int64_t y_overlap_end_shifted = y_overlap_end + l->ioffsety;
                const int64_t z_overlap_start_shifted = z_overlap_start + l->ioffsetz;
                const int64_t z_overlap_end_shifted = z_overlap_end + l->ioffsetz;

                const int64_t region_x_size = x_overlap_end_shifted - x_overlap_start_shifted;
                const int64_t region_y_size = y_overlap_end_shifted - y_overlap_start_shifted;
                const int64_t region_z_size = z_overlap_end_shifted - z_overlap_start_shifted;

                uint16_t *region = reader->second->load_region(
                    scale,
                    x_overlap_start_shifted, x_overlap_end_shifted,
                    y_overlap_start_shifted, y_overlap_end_shifted,
                    z_overlap_start_shifted, z_overlap_end_shifted);

                const int64_t cin = l->source_channel;
                const int64_t cout = l->target_channel;

                for (size_t i = x_overlap_start; i < x_overlap_end; i++)
                {
                    for (size_t j = y_overlap_start; j < y_overlap_end; j++)
                    {
                        for (size_t k = z_overlap_start; k < z_overlap_end; k++)
                        {
                            const int64_t i_s = i + l->ioffsetx;
                            const int64_t j_s = j + l->ioffsety;
                            const int64_t k_s = k + l->ioffsetz;

                            // Calculate the coordinates of the input and output inside their respective buffers
                            const size_t roffset = (cin * region_x_size * region_y_size * region_z_size) +
                                                   ((k_s - z_overlap_start_shifted) * region_y_size * region_x_size) +
                                                   ((j_s - y_overlap_start_shifted) * region_x_size) +
                                                   ((i_s - x_overlap_start_shifted));

                            const size_t ooffset = (cout * osizey * osizex * osizez) + // C
                                                   ((k - zs) * osizey * osizex) +      // Z
                                                   ((j - ys) * osizex) +               // Y
                                                   ((i - xs));                         // X

                            out_buffer[ooffset] = region[roffset];
                        }
                    }
                }

                free(region);
            }
        }

        if (CHUNK_TIMER)
        {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            size_t dt = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            std::cout << "Time difference = " << dt << " [us]" << std::endl;
        }

        return out_buffer;
    }

    // Returns false with a short reason in error when the write was refused
    // (nothing written) or when writing a chunk failed.
    bool replace_region(
        size_t scale,
        size_t xs, size_t xe,
        size_t ys, size_t ye,
        size_t zs, size_t ze,
        const char *data,
        std::string &error)
    {
        // Calculate size of output
        const size_t osizex = xe - xs;
        const size_t osizey = ye - ys;
        const size_t osizez = ze - zs;
        const size_t buffer_size = osizex * osizey * osizez * sizeof(uint16_t) * channel_count;

        // Define map for storing already decompressed chunks
        std::map<std::tuple<size_t, size_t, size_t, size_t, size_t>, region_chunk> chunk_cache;
        // Readers whose .meta mtime this request has checked
        std::set<packed_reader *> checked_readers;

        // Scaled metachunk size — clamp to >=1 so a thin axis (e.g.
        // z=1 with no Z pyramid) doesn't divide by zero further down.
        const size_t mcx = std::max<size_t>(1, mchunkx / scale);
        const size_t mcy = std::max<size_t>(1, mchunky / scale);
        const size_t mcz = std::max<size_t>(1, mchunkz / scale);

        // Variables to store chunk reader and data (shared in loop)
        packed_reader *chunk_reader = nullptr;
        std::tuple<size_t, size_t, size_t, size_t, size_t> *chunk_identifier = nullptr;
        size_t sub_chunk_id;
        uint16_t *chunk;

        // Variables for tracking the last chunks that were used
        size_t last_x, last_y, last_z, last_sub, last_c;
        size_t cxmin, cxmax, cxsize;
        size_t cymin, cymax, cysize;
        size_t czmin, czmax, czsize;

        // Nothing has been written when this runs: the first loop edits copies only
        auto reject = [&](const char *reason, const std::string &where) -> bool
        {
            std::string note;
            if (log_limit_write_refused.allow(note))
            {
                std::cerr << "Write refused (" << reason << "): " << fname << ' ' << where
                          << " box " << xs << '-' << xe << '_' << ys << '-' << ye << '_' << zs << '-' << ze << note << std::endl;
            }
            for (auto it = chunk_cache.begin(); it != chunk_cache.end(); it++)
            {
                free(it->second.ptr);
            }
            if (chunk_identifier != nullptr)
            {
                delete chunk_identifier;
            }
            error = reason;
            return false;
        };

        for (size_t c = 0; c < channel_count; c++)
        {
            for (size_t i = xs; i < xe; i++)
            {
                const size_t xmin = mcx * (i / mcx);                             // lower bound of mchunk
                const size_t xmax = std::min((size_t)xmin + mcx, (size_t)sizex); // upper bound of mchunk
                const size_t xsize = xmax - xmin;                                // size of mchunk
                const size_t chunk_id_x = i / ((size_t)mcx);                     // mchunk x id
                const size_t x_in_chunk = i - xmin;                              // x displacement inside chunk

                for (size_t j = ys; j < ye; j++)
                {
                    const size_t ymin = mcy * (j / mcy);
                    const size_t ymax = std::min((size_t)ymin + mcy, (size_t)sizey);
                    const size_t ysize = ymax - ymin;
                    const size_t chunk_id_y = j / ((size_t)mcy);
                    const size_t y_in_chunk = j - ymin;

                    for (size_t k = zs; k < ze; k++)
                    {
                        const size_t zmin = mcz * (k / mcz);
                        const size_t zmax = std::min((size_t)zmin + mcz, (size_t)sizez);
                        const size_t zsize = zmax - zmin;
                        const size_t chunk_id_z = k / ((size_t)mcz);
                        const size_t z_in_chunk = k - zmin;

                        bool force = false;
                        if (chunk_reader == nullptr ||
                            chunk_identifier == nullptr ||
                            last_x != chunk_id_x ||
                            last_y != chunk_id_y ||
                            last_z != chunk_id_z ||
                            last_c != c)
                        {
                            force = true;
                            chunk_reader = get_mchunk(scale, c, chunk_id_x, chunk_id_y, chunk_id_z);

                            if (chunk_reader == nullptr)
                            {
                                return reject("Missing mchunk", "mchunk " + std::to_string(chunk_id_x) + '_' + std::to_string(chunk_id_y) + '_' + std::to_string(chunk_id_z) + " channel " + std::to_string(c));
                            }

                            // Pick up a header rewritten on disk before any offset is computed from
                            // it, once per mchunk per request. A chunk this request covers entirely is
                            // never loaded, so the reload inside load_chunk would not run for it.
                            if (checked_readers.insert(chunk_reader).second)
                            {
                                chunk_reader->reload_if_modified();
                            }

                            last_x = chunk_id_x;
                            last_y = chunk_id_y;
                            last_z = chunk_id_z;
                        }

                        if (!chunk_reader->is_valid)
                        {
                            return reject("Unusable mchunk header", chunk_reader->meta_fname);
                        }

                        // Shift ranges for cropping
                        const size_t x_in_chunk_offset = x_in_chunk + chunk_reader->cropstartx;
                        const size_t y_in_chunk_offset = y_in_chunk + chunk_reader->cropstarty;
                        const size_t z_in_chunk_offset = z_in_chunk + chunk_reader->cropstartz;

                        if (x_in_chunk_offset >= chunk_reader->sizex ||
                            y_in_chunk_offset >= chunk_reader->sizey ||
                            z_in_chunk_offset >= chunk_reader->sizez)
                        {
                            return reject("Region outside stored tile", chunk_reader->meta_fname);
                        }

                        // Find sub chunk id from coordinates
                        sub_chunk_id = chunk_reader->find_index(x_in_chunk_offset, y_in_chunk_offset, z_in_chunk_offset);

                        // Only perform this step if there has been a change in chunk
                        if (force ||
                            last_sub != sub_chunk_id)
                        {
                            // Replace the chunk id with the new one
                            if (chunk_identifier != nullptr)
                            {
                                delete chunk_identifier;
                            }
                            chunk_identifier = new std::tuple(c, chunk_id_x, chunk_id_y, chunk_id_z, sub_chunk_id);

                            // Find the start/stop coordinates of this chunk
                            cxmin = ((size_t)chunk_reader->chunkx) * (x_in_chunk_offset / ((size_t)chunk_reader->chunkx)); // Minimum value of the chunk
                            cxmax = std::min((size_t)cxmin + chunk_reader->chunkx, (size_t)chunk_reader->sizex);           // Maximum value of the chunk
                            cxsize = cxmax - cxmin;                                                                        // Size of the chunk

                            cymin = ((size_t)chunk_reader->chunky) * (y_in_chunk_offset / ((size_t)chunk_reader->chunky));
                            cymax = std::min((size_t)cymin + chunk_reader->chunky, (size_t)chunk_reader->sizey);
                            cysize = cymax - cymin;

                            czmin = ((size_t)chunk_reader->chunkz) * (z_in_chunk_offset / ((size_t)chunk_reader->chunkz));
                            czmax = std::min((size_t)czmin + chunk_reader->chunkz, (size_t)chunk_reader->sizez);
                            czsize = czmax - czmin;

                            if (cxmax <= cxmin || cymax <= cymin || czmax <= czmin)
                            {
                                return reject("Region outside stored tile", chunk_reader->meta_fname);
                            }

                            // Check if the chunk is in the tmp cache
                            region_chunk &cached = chunk_cache[*chunk_identifier];
                            if (cached.ptr != nullptr && !cached.has_extent(cxmin, cymin, czmin, cxsize, cysize, czsize))
                            {
                                return reject("Chunk extent changed during the write", chunk_reader->meta_fname);
                            }
                            if (cached.ptr == nullptr)
                            {
                                // Where the request lies inside this mchunk's stored tile
                                const size_t rx0 = std::max(xs, xmin) - xmin + chunk_reader->cropstartx;
                                const size_t rx1 = std::min(xe, xmin + mcx) - xmin + chunk_reader->cropstartx;
                                const size_t ry0 = std::max(ys, ymin) - ymin + chunk_reader->cropstarty;
                                const size_t ry1 = std::min(ye, ymin + mcy) - ymin + chunk_reader->cropstarty;
                                const size_t rz0 = std::max(zs, zmin) - zmin + chunk_reader->cropstartz;
                                const size_t rz1 = std::min(ze, zmin + mcz) - zmin + chunk_reader->cropstartz;

                                if (rx0 <= cxmin && rx1 >= cxmax && ry0 <= cymin && ry1 >= cymax && rz0 <= czmin && rz1 >= czmax)
                                {
                                    // Every voxel of the chunk is overwritten, so what is stored
                                    // does not matter, and a chunk that cannot be read can be
                                    // replaced
                                    chunk = (uint16_t *)calloc(cxsize * cysize * czsize, sizeof(uint16_t));
                                    if (chunk == nullptr)
                                    {
                                        return reject("Out of memory", chunk_reader->data_fname + " chunk " + std::to_string(sub_chunk_id));
                                    }
                                }
                                else
                                {
                                    // Writing back a chunk that failed to load would replace its
                                    // voxels outside this region with zeros
                                    bool load_failed = false;
                                    chunk = chunk_reader->load_chunk(sub_chunk_id, cxsize, cysize, czsize, &load_failed);
                                    if (chunk == nullptr || load_failed)
                                    {
                                        free(chunk);
                                        return reject("Could not read existing chunk", chunk_reader->data_fname + " chunk " + std::to_string(sub_chunk_id));
                                    }
                                }
                                cached = region_chunk{chunk, cxmin, cymin, czmin, cxsize, cysize, czsize};
                            }
                            chunk = cached.ptr;

                            // Store this ID as the most recent chunk
                            last_sub = sub_chunk_id;
                            last_c = c;
                        }

                        // The chunk was sized before a reload moved this voxel outside it
                        if (x_in_chunk_offset - cxmin >= cxsize ||
                            y_in_chunk_offset - cymin >= cysize ||
                            z_in_chunk_offset - czmin >= czsize)
                        {
                            return reject("Chunk extent changed during the write", chunk_reader->meta_fname);
                        }

                        // Calculate the coordinates of the input and output inside their respective buffers
                        const size_t coffset = ((x_in_chunk_offset - cxmin) * cysize * czsize) + // X
                                               ((y_in_chunk_offset - cymin) * czsize) +          // Y
                                               (z_in_chunk_offset - czmin);                      // Z

                        const size_t ooffset = (c * osizey * osizex * osizez) + // C
                                               ((k - zs) * osizey * osizex) +   // Z
                                               ((j - ys) * osizex) +            // Y
                                               ((i - xs));                      // X

                        // out_buffer[ooffset] = chunk[coffset];
                        chunk[coffset] = ((uint16_t *)data)[ooffset];
                    }
                }
            }
        }

        if (chunk_identifier != nullptr)
        {
            delete chunk_identifier;
        }

        // TODO load chunks back
        bool all_written = true;
        for (auto it = chunk_cache.begin(); it != chunk_cache.end(); it++)
        {
            std::tuple<size_t, size_t, size_t, size_t, size_t> id_tuple = it->first;

            // std::tuple(c, chunk_id_x, chunk_id_y, chunk_id_z, sub_chunk_id);
            // chunk_reader = get_mchunk(scale, c, chunk_id_x, chunk_id_y, chunk_id_z);

            packed_reader *chunk_writer = get_mchunk(1, std::get<0>(id_tuple), std::get<1>(id_tuple), std::get<2>(id_tuple), std::get<3>(id_tuple));

            if (chunk_writer == nullptr)
            {
                all_written = false;
                free(it->second.ptr);
                continue;
            }

            size_t chunk_size = it->second.xsize * it->second.ysize * it->second.zsize * sizeof(uint16_t);

            if (!chunk_writer->overwrite_chunk(std::get<4>(id_tuple), it->second.ptr, chunk_size))
            {
                all_written = false;
            }
            free(it->second.ptr);
        }

        if (!all_written)
        {
            // The other chunks of the request may have been written
            std::string note;
            if (log_limit_write_failed.allow(note))
            {
                std::cerr << "Write failed: " << fname << " box " << xs << '-' << xe << '_' << ys << '-' << ye << '_' << zs << '-' << ze << note << std::endl;
            }
            error = "Could not write chunk";
        }
        return all_written;
    }

    void print_info()
    {
        std::cout << '[' << fname << "] " << " dtype=" << dtype << " channels=" << channel_count << " chunks=(" << mchunkx << ','
                  << mchunky << ',' << mchunkz << ") size=(" << sizex << ',' << sizey << ',' << sizez << ") "
                  << "tile_counts=(" << mcountx << ',' << mcounty << ',' << mcountz << ") "
                  << "scales=[";

        for (size_t n : scales)
            std::cout << n << ',';

        std::cout << "]" << std::endl;

        if (type == DESCRIPTOR)
        {
            for (size_t i = 0; i < descriptor_layers.size(); i++)
            {
                std::cout << "\t[layer " << i << "] from=\"" << descriptor_layers[i]->source_name << "\":" << descriptor_layers[i]->source_channel
                          << " ch=" << descriptor_layers[i]->target_channel
                          << " size=(" << descriptor_layers[i]->sizex << ", " << descriptor_layers[i]->sizey << ", " << descriptor_layers[i]->sizez << ")"
                          << " ooffset=(" << descriptor_layers[i]->ooffsetx << ", " << descriptor_layers[i]->ooffsety << ", " << descriptor_layers[i]->ooffsetz << ")"
                          << std::endl;
            }
        }
    }
};
