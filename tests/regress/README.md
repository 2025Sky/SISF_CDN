# Black-box regression harness

Runs two CDN images side by side on the same generated datasets, sends both the
same requests, and reports every response that differs.

```
pip install numpy zstd
python harness.py --baseline tuffr5/sisf_cdn@sha256:83d43adc... --baseline-platform linux/amd64 \
                  --candidate sisf_cdn:candidate --work /tmp/regress \
                  --allow expected_diffs.json --report /tmp/report.json
```

- `fixtures.py` writes the datasets. Its bytes match pySISF 0.4.1 and the
  portal's tiled writer (checked file by file when it was written).
- Reads: status code and SHA-256 of every body. Writes: the portal's
  read-merge-PATCH sequence on each server's own copy, then re-reads and a
  byte comparison of every file on disk.
- A server that dies is recorded as CRASH and restarted. Two crashes on the
  same request count as the same behaviour; every crash is listed either way.
  A server that stays up but does not answer in time is recorded as TIMEOUT.
- After the write and danger stages, `scenarios` runs the cases that kill,
  hang or lose data on production, each on a dataset it builds while the
  server runs, with a liveness check after each:
  - s1: stale archive geometry after an in-place re-conversion (the
    2026-08-30 abort), then `/info` and a full read of the same dataset
    afterwards;
  - s2: a corrupt zstd frame; s3: an unknown compression type in an mchunk
    header;
  - s4: a writable layer with a missing `.data` file (production leaks its
    write lock; this case uses a 20 s client timeout and restarts a server
    that timed out);
  - s5: a PATCH over a chunk whose data is cut short (production overwrites
    the chunk's other voxels with zeros and answers 200);
  - s6: the portal's read-merge-PATCH with one mchunk's `.data` unreadable
    during the read only (renamed inside the container with `docker exec`
    and back), plus a viewer's read of the same failure;
  - s7: a tile re-converted in place with a wider tile while the server
    holds readers for it, then read and written in the one chunk whose
    extent grew; it runs last because production dies in it;
  - s8: a `raw_access` read wider than the mchunk's stored tile;
  - s9: s6's sequence with the `.meta` unreadable instead, on an mchunk the
    server has never opened, so no reader can be built for it during the
    read;
  - s10: an mchunk header naming video compression over zstd frames shorter
    than the video header, read by a viewer and by the portal and under a
    PATCH;
  - s11, s12: a dataset whose `metadata.bin` is empty, or names an mchunk
    size of 0 (production divides by it in the inventory scan; the harness
    removes the dataset before restarting a server that died on it);
  - s17: reads sent one after another on one keep-alive connection, as
    the portal's httpx pool and browsers send them: an image read over
    1 MiB, then a mesh file (a route that adds two headers), and the
    reverse; then GETs sent with `Expect: 100-continue` mixed with large
    and small reads, the last with `Connection: close`; then 16 threads,
    each on its own connection, cycling through large and small reads (448
    in all), every body checked against the same read on a fresh
    connection. After a body of 1 MiB or more, production's crow leaves a
    stale `connection` header in the connection's next response, which
    then carries two `Connection` headers, and frees it while that
    response is still being sent. Its `100 Continue` shares the completion
    of a whole response, which clears the real answer while it is still
    queued: the first body byte goes out as 0, and a `Connection: close`
    answer is cut off. The sanitizer build aborts on both;
  - s18: `skeleton_api` upload, replace and delete (a GET) on a dataset
    with a `traces.sql`, with `ls` and `get` before and after, on this
    server (writes off by default), on a second server with
    `SKELETON_API_WRITES=1` and its own dataset, and on one with a value
    that is not 0 or 1. Rows carry the time they were written, so the
    file's bytes are not compared; each server records whether it changed;
  - s13: `+channel=N` reads of the 3-channel fixture, each checked against
    channel N's block of the same read without the filter (boxes, a plane,
    a coarser level, a projection, a gaussian filter), and channels that do
    not exist.
- `expected_diffs.json` lists the differences that are intended, each with a
  reason and, under `expect`, the candidate's answer: any of `status`,
  `len`, `sha256`, `text` (or `text_prefix`, the start of the text) for a
  response, `sha256` for a file, or `"unchanged"` for a file that must still
  equal the dataset as built. An entry whose `expect` is missing or empty,
  or pins only the status of a 200 that has a body, fails the run unless it
  carries `"unpinned_reason"` saying why the answer cannot be pinned; the
  run lists those. `expect_baseline` pins the baseline's answer the same
  way where it is deterministic, e.g. a crash that shows the case reaches
  the production bug. Anything not listed fails the run, and so does any
  candidate CRASH or TIMEOUT (even where the baseline crashes the same way),
  a listed difference that did not occur (the candidate behaves like the
  baseline there again), and a listed difference where an answer is not the
  expected one.
- Each server's full log is saved to `<work>/<role>.log`, and s18's other
  servers' to `<work>/<role>-skeleton-<value>.log`.

Compare like with like: run both images on amd64. An integer division by zero
that kills the process on x86-64 returns 0 on arm64, so an arm64 candidate can
pass a case that crashes in production.
