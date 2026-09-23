"""Black-box regression harness: run two CDN images on identical data, send
both the same requests, and report every response that differs.

    python harness.py --baseline IMAGE --candidate IMAGE --work DIR
                      [--baseline-platform linux/amd64] [--candidate-platform ...]
                      [--candidate-env KEY=VALUE ...]
                      [--allow expected_diffs.json] [--report report.json]

Reads are compared by status code and SHA-256 of the body. Writes are sent to
each server's own copy of the data in the same order the SAVAII portal sends
them (read a 128-aligned box, merge labels, PATCH it back), then compared both
through re-reads and byte-for-byte on disk. A server that dies is recorded as
CRASH for that request, restarted, and the run continues; one that stays up
but does not answer in time is recorded as TIMEOUT.

Exit status: 0 when every difference is listed in the allow file, every
difference the allow file lists actually occurred, the candidate gave the
answer each listed difference's entry pins (an entry that pins nothing, or
only the status of a 200 with a body, needs an "unpinned_reason"), and the
candidate never crashed or timed out; 1 otherwise. A candidate crash fails the
run even where the baseline crashes too, because two crashes compare as equal,
and a listed difference that did not occur means the candidate behaves like
the baseline there again.
"""

import argparse
import gzip
import hashlib
import http.client
import json
import os
import shutil
import sqlite3
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np

import fixtures

READ_DATASETS = ["vol1c", "vol3c", "tiled2d", "tiled3d", "plane2d", "seg_empty", "seg_prefilled", "seg_holes"]
NG_CHUNKS = [(256, 256, 1), (256, 1, 256), (1, 256, 256), (32, 32, 32)]
FULL_READ_LIMIT = 4_000_000  # voxels x channels; above this only boxes are read
RES = (650, 650, 1500)
SCENARIO_TIMEOUT = 20  # s; a write stuck on a leaked lock never answers


class Server:
    def __init__(self, role, image, platform, data_dir, env=None):
        self.role, self.image, self.platform, self.data_dir = role, image, platform, data_dir
        self.env = dict(env or {})
        self.name = f"cdn-regress-{role}-{os.getpid()}"
        self.port = None
        # SHA-256 of each file of a scenario dataset as built, before any
        # request, keyed like disk_hashes(); "unchanged" in the allow file
        # compares against these
        self.fixture_hashes = {}

    def snapshot(self, ds):
        for dirpath, _, files in os.walk(os.path.join(self.data_dir, ds)):
            for f in files:
                path = os.path.join(dirpath, f)
                self.fixture_hashes[os.path.relpath(path, self.data_dir)] = file_sha(path)

    def exec(self, *args):
        """Runs a command inside the container, which sees the data at /data."""
        subprocess.run(["docker", "exec", self.name, *args], check=True, capture_output=True)

    def start(self):
        cmd = ["docker", "run", "-d", "--name", self.name, "-p", "127.0.0.1::6000",
               "-v", f"{self.data_dir}:/data"]
        for k, v in self.env.items():
            cmd += ["-e", f"{k}={v}"]
        if self.platform:
            cmd += ["--platform", self.platform]
        subprocess.run(cmd + [self.image], check=True, capture_output=True)
        self._wait()

    def restart(self):
        subprocess.run(["docker", "start", self.name], check=True, capture_output=True)
        self._wait()

    def restart_running(self):
        """For a server that is up but wedged (e.g. a thread stuck on a lock)."""
        subprocess.run(["docker", "restart", "-t", "5", self.name], check=True, capture_output=True)
        self._wait()

    def _wait(self):
        out = subprocess.run(["docker", "port", self.name, "6000/tcp"], check=True,
                             capture_output=True, text=True).stdout
        self.port = int(out.strip().splitlines()[0].rsplit(":", 1)[1])
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/version", timeout=5) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(1)
        raise RuntimeError(f"{self.role} did not answer /version within 300 s")

    def running(self):
        out = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}} {{.State.ExitCode}}", self.name],
                             capture_output=True, text=True).stdout.split()
        return out[0] == "true", int(out[1]) if len(out) > 1 else None

    def logs_tail(self, n=15):
        r = subprocess.run(["docker", "logs", "--tail", str(n), self.name], capture_output=True, text=True)
        return (r.stdout + r.stderr).strip().splitlines()

    def save_logs(self, path):
        with open(path, "w") as f:
            subprocess.run(["docker", "logs", self.name], stdout=f, stderr=subprocess.STDOUT)

    def stop(self):
        subprocess.run(["docker", "stop", "-t", "5", self.name], capture_output=True)

    def remove(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)

    def request(self, method, path, body=None, timeout=120, before_restart=None):
        """Returns (status, body bytes). status is 'CRASH' if the server died,
        'TIMEOUT' if it is alive but did not answer within timeout seconds,
        'NOCONN' if it is alive but the connection failed otherwise.
        before_restart runs after a crash and before the restart, e.g. to
        remove a dataset that would kill the restarted server's startup scan."""
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=body, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except Exception as e:  # connection reset, refused, timeout
            return self._failed(e, timeout, before_restart)

    def get_encoded(self, path, accept_encoding, timeout=120):
        """A GET that sends accept_encoding as its Accept-Encoding header, or
        no such header when it is None (urllib always sends "identity").
        Returns (status, headers with lower-case names, body as sent);
        status is 'CRASH', 'TIMEOUT' or 'NOCONN' as for request()."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            conn.putrequest("GET", path, skip_accept_encoding=True)
            if accept_encoding is not None:
                conn.putheader("Accept-Encoding", accept_encoding)
            conn.endheaders()
            r = conn.getresponse()
            return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
        except Exception as e:
            status, text = self._failed(e, timeout, None)
            return status, {}, text
        finally:
            conn.close()

    def get_keepalive(self, reqs, timeout=120):
        """Sends each (path, accept_encoding[, extra]) of reqs as a GET on ONE
        HTTP/1.1 connection, each after the answer before it was read in full;
        an accept_encoding of None sends no such header, and extra is a tuple
        of (name, value) headers to add (http.client reads and skips a
        100 Continue before an answer). Returns one
        (status, headers, body as sent, reused) per request sent, where
        headers is the list of (lower-case name, value) pairs, so a repeated
        header shows twice, and reused says whether the request went out on
        the connection the first one opened (http.client opens a new one
        silently when the server has closed it). After a failure the list
        ends with that request's ('CRASH' | 'TIMEOUT' | 'NOCONN', [], text,
        reused) and the rest are not sent."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        out, first = [], None
        try:
            for path, accept_encoding, *extra in reqs:
                reused = True
                try:
                    conn.putrequest("GET", path, skip_accept_encoding=True)
                    if accept_encoding is not None:
                        conn.putheader("Accept-Encoding", accept_encoding)
                    for name, value in (extra[0] if extra else ()):
                        conn.putheader(name, value)
                    conn.endheaders()
                    first = first or conn.sock
                    reused = conn.sock is first
                    r = conn.getresponse()
                    out.append((r.status, [(k.lower(), v) for k, v in r.getheaders()], r.read(), reused))
                except Exception as e:
                    status, text = self._failed(e, timeout, None)
                    out.append((status, [], text, reused))
                    break
        finally:
            conn.close()
        return out

    def _failed(self, e, timeout, before_restart):
        time.sleep(1)
        alive, code = self.running()
        if not alive:
            tail = self.logs_tail()
            if before_restart is not None:
                before_restart()
            self.restart()
            return "CRASH", f"exit={code}; {' | '.join(tail[-4:])}; {type(e).__name__}"
        if isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None), TimeoutError):
            return "TIMEOUT", f"no response within {timeout} s"
        return "NOCONN", f"{type(e).__name__}: {e}"


def file_sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def digest(status, body):
    if isinstance(body, str):
        return {"status": status, "len": None, "sha256": None, "text": body[:300]}
    text = body[:200].decode("utf-8", "replace") if not _binary(body) else None
    return {"status": status, "len": len(body), "sha256": hashlib.sha256(body).hexdigest(), "text": text}


def _binary(b):
    return b"\x00" in b[:200]


def box(x0, x1, y0, y1, z0, z1):
    return f"{x0}-{x1}_{y0}-{y1}_{z0}-{z1}"


def read_corpus(info_by_ds):
    """Deterministic list of (id, method, path). Built from the BASELINE's /info
    so both servers are asked exactly the same thing."""
    reqs = [("GET /", "GET", "/"), ("GET /version", "GET", "/version"),
            ("GET /inventory", "GET", "/inventory"),
            ("GET /nope/info", "GET", "/nope/info"), ("GET /nope/1/box", "GET", "/nope/1/0-1_0-1_0-1")]
    for ds, info in info_by_ds.items():
        reqs.append((f"GET /{ds}/info", "GET", f"/{ds}/info"))
        reqs.append((f"GET /{ds}/provenance", "GET", f"/{ds}/provenance"))
        if info is None:
            continue
        nch = int(info.get("num_channels", 1))
        rng = np.random.default_rng(int(hashlib.sha256(ds.encode()).hexdigest()[:8], 16))
        for sc in info["scales"]:
            key, (X, Y, Z) = sc["key"], sc["size"]
            p = f"/{ds}/{key}/"
            add = lambda tag, b: reqs.append((f"GET {p}{b} [{tag}]", "GET", p + b))
            if X * Y * Z * nch <= FULL_READ_LIMIT:
                add("full", box(0, X, 0, Y, 0, Z))
            for z in sorted({0, Z // 2, Z - 1}):
                add("xy", box(0, X, 0, Y, z, z + 1))
            add("xz", box(0, X, Y // 2, Y // 2 + 1, 0, Z))
            add("yz", box(X // 2, X // 2 + 1, 0, Y, 0, Z))
            for cs in NG_CHUNKS:
                n = [max(1, -(-d // c)) for d, c in zip((X, Y, Z), cs)]
                for idx in sorted({(0, 0, 0), tuple(v // 2 for v in n), tuple(v - 1 for v in n)}):
                    lo = [i * c for i, c in zip(idx, cs)]
                    hi = [min((i + 1) * c, d) for i, c, d in zip(idx, cs, (X, Y, Z))]
                    add(f"ng{cs}", box(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
            for i in range(12):
                x0, y0, z0 = (int(rng.integers(0, d)) for d in (X, Y, Z))
                x1, y1, z1 = (int(rng.integers(a + 1, d + 1)) for a, d in ((x0, X), (y0, Y), (z0, Z)))
                add(f"rand{i}", box(x0, x1, y0, y1, z0, z1))
            # Out-of-range and malformed requests: both servers must answer alike.
            add("beyond-x", box(0, X + 50, 0, min(Y, 8), 0, 1))
            add("beyond-z", box(0, min(X, 8), 0, min(Y, 8), max(0, Z - 1), Z + 5))
            add("empty", box(5, 5, 0, 8, 0, 1))
            add("reversed", box(8, 2, 0, 8, 0, 1))
            add("two-axes", "0-8_0-8")
            add("letters", "a-b_c-d_e-f")
        # Scales the dataset does not have. At 1024 every axis downsamples below
        # one voxel: upstream #34 clamps that size to 1, so the one-voxel box is
        # the probe for that known difference.
        for key in ("3", "0", "1024"):
            reqs.append((f"GET /{ds}/{key}/small [scale {key}]", "GET", f"/{ds}/{key}/" + box(0, 4, 0, 4, 0, 1)))
        reqs.append((f"GET /{ds}/1024/one-voxel [scale 1024]", "GET", f"/{ds}/1024/" + box(0, 1, 0, 1, 0, 1)))
        reqs.append((f"GET /{ds}+token=wrong/1/small", "GET", f"/{ds}+token=wrong/1/" + box(0, 4, 0, 4, 0, 1)))
        reqs.append((f"GET /{ds}+token=right/1/small", "GET",
                     f"/{ds}+token={fixtures.TOKEN}/1/" + box(0, 4, 0, 4, 0, 1)))
    for pc in ("pc_new", "pc_old", "pc_nodims", "pc_missing"):
        reqs.append((f"GET pointcloud {pc} info", "GET", f"/vol1c/pointcloud/{pc}/info"))
        reqs.append((f"GET pointcloud {pc} spatial0", "GET", f"/vol1c/pointcloud/{pc}/spatial0/0_0_0"))
    return reqs


def write_plan():
    """(id, dataset, kind, box, seed). Boxes are portal-shaped: 128-aligned
    super-chunks clamped to the dataset, which straddle the 96/48 mchunks."""
    return [
        ("w01 merge sc(0,0,0)", "seg_empty", "merge", (0, 128, 0, 128, 0, 64), 1),
        ("w02 merge sc(1,1,1) edge", "seg_empty", "merge", (128, 160, 128, 150, 64, 70), 2),
        ("w03 merge sc(1,0,0)", "seg_empty", "merge", (128, 160, 0, 128, 0, 64), 3),
        ("w04 raw unaligned", "seg_empty", "raw", (10, 50, 20, 70, 5, 40), 4),
        ("w05 erase", "seg_empty", "erase", (30, 100, 30, 100, 10, 30), 5),
        ("w06 merge overwrite sc(0,0,0)", "seg_empty", "merge", (0, 128, 0, 128, 0, 64), 6),
        ("w07 merge single plane", "seg_empty", "merge", (0, 128, 0, 128, 33, 34), 7),
        ("w08 merge prefilled", "seg_prefilled", "merge", (0, 100, 0, 90, 0, 37), 8),
        ("w09 erase prefilled", "seg_prefilled", "erase", (20, 40, 20, 40, 0, 37), 9),
    ]


REFUSAL_PLAN = [
    # (id, path, body voxels or None for a correct-length body)
    ("r01 wrong token", "/seg_empty+token=wrong/write/1/" + box(0, 32, 0, 32, 0, 32), None),
    ("r02 no token", "/seg_empty/write/1/" + box(0, 32, 0, 32, 0, 32), None),
    ("r03 image layer (no .sisf_access)", f"/vol1c+token={fixtures.TOKEN}/write/1/" + box(0, 32, 0, 32, 0, 32), None),
    ("r04 multi-channel layer", f"/vol3c+token={fixtures.TOKEN}/write/1/" + box(0, 32, 0, 32, 0, 20), None),
    ("r05 malformed box", f"/seg_empty+token={fixtures.TOKEN}/write/1/0-32_0-32", None),
    ("r06 unknown dataset", f"/nope+token={fixtures.TOKEN}/write/1/" + box(0, 32, 0, 32, 0, 32), None),
]

# Requests that may crash or corrupt a server. Run last, one at a time, with a
# liveness check after each; a crash is a result, not a harness failure.
DANGER_PLAN = [
    ("d01 body too short", f"/seg_empty+token={fixtures.TOKEN}/write/1/" + box(0, 32, 0, 32, 0, 32), 32 * 32 * 16),
    ("d02 body too long", f"/seg_empty+token={fixtures.TOKEN}/write/1/" + box(0, 32, 0, 32, 0, 32), 32 * 32 * 64),
    ("d03 box beyond size", f"/seg_empty+token={fixtures.TOKEN}/write/1/" + box(150, 190, 0, 32, 0, 32), None),
    ("d04 scale 2", f"/seg_empty+token={fixtures.TOKEN}/write/2/" + box(0, 32, 0, 32, 0, 32), None),
    ("d05 empty box", f"/seg_empty+token={fixtures.TOKEN}/write/1/" + box(5, 5, 0, 32, 0, 32), 0),
    # Straddles present mchunk (0,0,0) and the missing (1,0,0). Production
    # dereferences the null reader for the missing one and dies (exit 139).
    ("d06 write across missing mchunk", f"/seg_holes+token={fixtures.TOKEN}/write/1/" + box(64, 128, 0, 32, 0, 32),
     64 * 32 * 32),
]


def labels_for(shape_zyx, seed):
    z, y, x = np.indices(shape_zyx)
    val = ((x * 3 + y * 5 + z * 7 + seed) % 5 + 1).astype(np.uint16)
    mask = ((x + y + z + seed) % 3) == 0
    return val, mask


def run_writes(server, results):
    for wid, ds, kind, (x0, x1, y0, y1, z0, z1), seed in write_plan():
        shape = (z1 - z0, y1 - y0, x1 - x0)
        b = box(x0, x1, y0, y1, z0, z1)
        if kind == "merge":
            st, cur = server.request("GET", f"/{ds}+token={fixtures.TOKEN}/1/{b}")
            results[f"{wid} (read)"] = digest(st, cur)
            if st != 200:
                continue
            vol = np.frombuffer(cur, dtype=np.uint16).reshape(shape).copy()
            val, mask = labels_for(shape, seed)
            vol[mask] = val[mask]
        elif kind == "raw":
            vol = labels_for(shape, seed)[0]
        else:
            vol = np.zeros(shape, dtype=np.uint16)
        st, body = server.request("PATCH", f"/{ds}+token={fixtures.TOKEN}/write/1/{b}", vol.tobytes())
        results[wid] = digest(st, body)
    for rid, path, nvox in REFUSAL_PLAN:
        results[rid] = digest(*server.request("PATCH", path, np.ones(32 * 32 * 32, dtype=np.uint16).tobytes()))


def run_danger(server, results):
    for did, path, nvox in DANGER_PLAN:
        n = 32 * 32 * 32 if nvox is None else nvox
        results[did] = digest(*server.request("PATCH", path, np.full(n, 3, dtype=np.uint16).tobytes()))
        alive, code = server.running()
        results[f"{did} (alive after)"] = {"status": "alive" if alive else f"dead exit={code}",
                                           "len": None, "sha256": None, "text": None}
        if not alive:
            server.restart()


def sc_stale_geometry(server, results, sid):
    """The 2026-08-30 production abort: a tiled dataset is served, re-converted
    in place with a larger overlap (the CDN keeps the old tile step from
    metadata.bin), then an alignment moves one tile's crop start in x and y to
    the top of its new margin, and that tile is read."""
    ds, tile = "sc_stale", (96, 96, 1)
    root = os.path.join(server.data_dir, ds)
    fixtures.tiled(root, 1, (3, 2, 1), tile, (24, 24, 0), RES, 3)  # step 72
    time.sleep(1.2)  # production compares .meta mtimes in whole seconds
    results[f"{sid}: /info before"] = digest(*server.request("GET", f"/{ds}/info"))
    results[f"{sid}: read before"] = digest(*server.request("GET", f"/{ds}/1/" + box(0, 216, 0, 144, 0, 1)))
    fixtures.tiled(root, 1, (3, 2, 1), tile, (32, 32, 0), RES, 3)  # in place, step 64
    time.sleep(1.2)
    results[f"{sid}: read after re-conversion"] = digest(*server.request("GET", f"/{ds}/1/" + box(0, 192, 0, 128, 0, 1)))
    margin = 32
    for scale in ("1X", "2X"):
        p = os.path.join(root, "meta", f"chunk_1_0_0.0.{scale}.meta")
        if not os.path.exists(p):
            continue
        f_ = 1 if scale == "1X" else 2
        with open(p, "r+b") as f:
            v = list(struct.unpack(fixtures.SHARD_HEADER_LAYOUT, f.read(86)))
            w = v[11] - v[10]; v[10] = margin // f_; v[11] = v[10] + w
            h = v[13] - v[12]; v[12] = margin // f_; v[13] = v[12] + h
            f.seek(0)
            f.write(struct.pack(fixtures.SHARD_HEADER_LAYOUT, *v))
    time.sleep(1.2)
    results[f"{sid}: read aligned tile"] = digest(*server.request("GET", f"/{ds}/1/" + box(72, 144, 0, 72, 0, 1)))
    # The archive geometry (tile step, size) is read once per process. A
    # server that died above comes back with the new one; one that stayed up
    # keeps the old one until it is restarted.
    results[f"{sid}: /info after the event"] = digest(*server.request("GET", f"/{ds}/info"))
    results[f"{sid}: full read after the event"] = digest(*server.request("GET", f"/{ds}/1/" + box(0, 192, 0, 128, 0, 1)))


def _vol(root):
    fixtures.untiled(root, fixtures.pattern((1, 100, 90, 37), 1), (64, 64, 32), RES, 2)


def sc_corrupt_zstd(server, results, sid):
    """One chunk's zstd frame has a broken magic number."""
    ds = "sc_zstd"
    root = os.path.join(server.data_dir, ds)
    _vol(root)
    with open(os.path.join(root, "meta", "chunk_0_0_0.0.1X.meta"), "rb") as f:
        f.seek(86 + 12 * 1)  # chunk 1: x 0-32, y 32-64, z 0-32
        offset, size = struct.unpack(fixtures.SHARD_LINE_LAYOUT, f.read(12))
    with open(os.path.join(root, "data", "chunk_0_0_0.0.1X.data"), "r+b") as f:
        f.seek(offset)
        f.write(b"\0\0\0\0")
    results[f"{sid}: read"] = digest(*server.request("GET", f"/{ds}/1/" + box(0, 64, 0, 64, 0, 32)))


def sc_bad_compression(server, results, sid):
    """An mchunk header names a compression type that does not exist."""
    ds = "sc_comp"
    root = os.path.join(server.data_dir, ds)
    _vol(root)
    with open(os.path.join(root, "meta", "chunk_0_0_0.0.1X.meta"), "r+b") as f:
        f.seek(6)  # version, dtype, channels, compression
        f.write(struct.pack("<H", 9))
    results[f"{sid}: read"] = digest(*server.request("GET", f"/{ds}/1/" + box(0, 64, 0, 64, 0, 32)))


def sc_missing_data(server, results, sid):
    """A writable layer whose .data file for mchunk (0,0,0) is gone (its .meta
    is still there). Production answers the first PATCH with 200 and leaks the
    global chunk lock, so the next PATCH anywhere hangs."""
    ds = "sc_nodata"
    root = os.path.join(server.data_dir, ds)
    fixtures.segmentation(root, (160, 150, 70), (96, 96, 48), RES)
    os.remove(os.path.join(root, "data", "chunk_0_0_0.0.1X.data"))
    body = np.full(32 * 32 * 32, 3, dtype=np.uint16).tobytes()
    w = f"/{ds}+token={fixtures.TOKEN}/write/1/"
    results[f"{sid}: PATCH into it"] = digest(*server.request("PATCH", w + box(0, 32, 0, 32, 0, 32), body,
                                                             timeout=SCENARIO_TIMEOUT))
    results[f"{sid}: second PATCH elsewhere"] = digest(*server.request("PATCH", w + box(96, 128, 0, 32, 0, 32), body,
                                                                      timeout=SCENARIO_TIMEOUT))
    if any(results[k]["status"] == "TIMEOUT" for k in results if k.startswith(sid)):
        server.restart_running()


def sc_unreadable_chunk_write(server, results, sid):
    """A prefilled writable layer whose .data is cut short inside the last
    chunk of mchunk (0,0,0), then a PATCH covering part of that chunk.
    Production reads the chunk as zeros, merges the request into them and
    writes the result back with 200, so the chunk's other voxels become 0."""
    ds = "sc_short"
    root = os.path.join(server.data_dir, ds)
    fixtures.segmentation(root, (100, 90, 37), (64, 64, 32), RES, prefill_seed=5)
    with open(os.path.join(root, "meta", "chunk_0_0_0.0.1X.meta"), "rb") as f:
        f.seek(86 + 12 * 3)  # chunk 3: x 32-64, y 32-64, z 0-32, the last frame in the file
        offset, size = struct.unpack(fixtures.SHARD_LINE_LAYOUT, f.read(12))
    with open(os.path.join(root, "data", "chunk_0_0_0.0.1X.data"), "r+b") as f:
        f.truncate(offset + size // 2)
    server.snapshot(ds)
    body = np.full(16 * 16 * 16, 3, dtype=np.uint16).tobytes()
    results[f"{sid}: PATCH part of the chunk"] = digest(*server.request(
        "PATCH", f"/{ds}+token={fixtures.TOKEN}/write/1/" + box(40, 56, 40, 56, 0, 16), body))
    results[f"{sid}: read the chunk back"] = digest(*server.request("GET", f"/{ds}/1/" + box(32, 64, 32, 64, 0, 32)))


def seg_labels_zyx(size, mchunk, seed):
    """The labels fixtures.segmentation(prefill_seed=seed) writes, as a full
    volume in the order the CDN returns it (z, y, x)."""
    vol = np.zeros(size, dtype=np.uint16)
    for i, (i0, i1) in enumerate(fixtures.iterate_bounded(size[0], mchunk[0])):
        for j, (j0, j1) in enumerate(fixtures.iterate_bounded(size[1], mchunk[1])):
            for k, (k0, k1) in enumerate(fixtures.iterate_bounded(size[2], mchunk[2])):
                shape = (i1 - i0, j1 - j0, k1 - k0)
                vol[i0:i1, j0:j1, k0:k1] = fixtures.pattern(shape, seed + i * 9 + j * 5 + k) % 7
    return vol.transpose(2, 1, 0)


def sc_read_failure_before_patch(server, results, sid):
    """The portal's read-merge-PATCH over a prefilled writable layer, with
    mchunk (0,0,0)'s .data unreadable for the read only (renamed inside the
    container and back). Production answers that read with zeros for the
    mchunk, and the PATCH writes them over its labels, 200 both times. The
    portal's read carries its token; a viewer's does not."""
    _read_failure_before_patch(server, results, sid, "sc_strict", "data")


def sc_cold_meta_before_patch(server, results, sid):
    """The same sequence with mchunk (0,0,0)'s .meta unreadable instead, on
    a server that has never opened that mchunk: no reader can be built for
    it during the read, so it reads as if the mchunk were missing, and the
    PATCH, which finds the .meta again, writes the zeros over its labels on
    production."""
    _read_failure_before_patch(server, results, sid, "sc_cold", "meta")


def _read_failure_before_patch(server, results, sid, ds, ext):
    size, mchunk, seed = (96, 64, 32), (64, 64, 32), 7
    root = os.path.join(server.data_dir, ds)
    fixtures.segmentation(root, size, mchunk, RES, prefill_seed=seed)
    server.snapshot(ds)
    b, shape = box(0, 96, 0, 64, 0, 32), (32, 64, 96)  # the portal's clamped super-chunk
    path = f"/data/{ds}/{ext}/chunk_0_0_0.0.1X.{ext}"
    server.exec("mv", path, path + ".away")
    results[f"{sid}: viewer read (no token)"] = digest(*server.request("GET", f"/{ds}/1/{b}"))
    st, cur = server.request("GET", f"/{ds}+token={fixtures.TOKEN}/1/{b}")
    results[f"{sid}: portal read before the PATCH"] = digest(st, cur)
    server.exec("mv", path + ".away", path)
    if st == 200:
        # The portal only goes on after a 200. A stroke in mchunk (1,0,0),
        # painted where nothing is stored yet (protect_existing).
        vol = np.frombuffer(cur, dtype=np.uint16).reshape(shape).copy()
        stroke = vol[0:10, 20:30, 70:80]
        stroke[stroke == 0] = 9
        results[f"{sid}: PATCH"] = digest(*server.request(
            "PATCH", f"/{ds}+token={fixtures.TOKEN}/write/1/{b}", vol.tobytes()))
    else:
        results[f"{sid}: PATCH"] = {"status": "not sent", "len": None, "sha256": None, "text": "the read failed"}
    st, after = server.request("GET", f"/{ds}/1/{b}")
    results[f"{sid}: read back"] = digest(st, after)
    if st == 200:
        truth = seg_labels_zyx(size, mchunk, seed)
        got = np.frombuffer(after, dtype=np.uint16).reshape(shape)
        lost = int(((truth != 0) & (got != truth)).sum())
        results[f"{sid}: stored labels lost"] = {"status": "count", "len": None, "sha256": None, "text": str(lost)}


def _set_crop_start(meta, x0, y0, width=72):
    """An alignment: move a tile's crop window, keeping its width."""
    with open(meta, "r+b") as f:
        v = list(struct.unpack(fixtures.SHARD_HEADER_LAYOUT, f.read(86)))
        v[10], v[11], v[12], v[13] = x0, x0 + width, y0, y0 + width
        f.seek(0)
        f.write(struct.pack(fixtures.SHARD_HEADER_LAYOUT, *v))


def _tiled_skeleton(root, tile, crop_start):
    """A writable 3x2 tiled layer with nothing written yet, tile step 72,
    1X only."""
    fixtures.write_metadata(root, 1, (72, 72, 1), RES, (216, 144, 1))
    for tx in range(3):
        for ty in range(2):
            name = f"chunk_{tx}_{ty}_0.0.1X"
            fixtures.write_skeleton(f"{root}/data/{name}.data", f"{root}/meta/{name}.meta", tile, (32, 32, 1))
            _set_crop_start(f"{root}/meta/{name}.meta", *crop_start)
    with open(f"{root}/.sisf_access", "w") as f:
        f.write(fixtures.TOKEN + "\n")


def sc_regrown_tile(server, results, sid):
    """A tiled dataset re-converted in place with a wider tile (88 -> 96 px,
    so the last chunk in x grows from 24 to 32 px) while the server holds
    readers for the old tiles, then tile (1,0,0)'s crop start moved from 12
    to 24. The box read and written below stays in one chunk under the new
    geometry, so a reader that reloads inside that chunk's first load keeps
    the old, smaller chunk extent for the rest of the request. Production
    dies on the read (the new frame does not fit the old buffer)."""
    ds, dsw = "sc_regrow", "sc_regrow_w"
    root, rootw = os.path.join(server.data_dir, ds), os.path.join(server.data_dir, dsw)
    old_tile, new_tile = (88, 96, 1), (96, 96, 1)
    fixtures.tiled(root, 1, (3, 2, 1), old_tile, (16, 24, 0), RES, 5)  # step 72, crop start (8, 12)
    _set_crop_start(os.path.join(root, "meta", "chunk_1_0_0.0.1X.meta"), 12, 12)
    _tiled_skeleton(rootw, old_tile, (8, 12))
    _set_crop_start(os.path.join(rootw, "meta", "chunk_1_0_0.0.1X.meta"), 12, 12)
    time.sleep(1.2)
    results[f"{sid}: read before"] = digest(*server.request("GET", f"/{ds}/1/" + box(0, 216, 0, 144, 0, 1)))
    results[f"{sid}: writable copy: read before"] = digest(*server.request(
        "GET", f"/{dsw}/1/" + box(0, 216, 0, 144, 0, 1)))
    fixtures.tiled(root, 1, (3, 2, 1), new_tile, (24, 24, 0), RES, 5)  # in place, step 72, crop start (12, 12)
    _set_crop_start(os.path.join(root, "meta", "chunk_1_0_0.0.1X.meta"), 24, 24)
    _tiled_skeleton(rootw, new_tile, (12, 12))
    _set_crop_start(os.path.join(rootw, "meta", "chunk_1_0_0.0.1X.meta"), 24, 24)
    time.sleep(1.2)
    b = box(124, 144, 0, 8, 0, 1)
    results[f"{sid}: read across the regrown chunk"] = digest(*server.request("GET", f"/{ds}/1/{b}"))
    results[f"{sid}: same read again"] = digest(*server.request("GET", f"/{ds}/1/{b}"))
    results[f"{sid}: PATCH the writable copy"] = digest(*server.request(
        "PATCH", f"/{dsw}+token={fixtures.TOKEN}/write/1/{b}", np.full(20 * 8, 3, dtype=np.uint16).tobytes()))
    results[f"{sid}: writable copy: read the box back"] = digest(*server.request("GET", f"/{dsw}/1/{b}"))


def sc_short_video_frame(server, results, sid):
    """An mchunk header that names video compression (type 2) over zstd
    frames shorter than the video header decode_stack_native reads (13
    uint32 fields and a uint64). A constant chunk compresses to about 20
    bytes. Read by a viewer, by the portal (token) and under a PATCH of part
    of a chunk; production reads past the frame and dies each time."""
    ds = "sc_video"
    root = os.path.join(server.data_dir, ds)
    fixtures.write_metadata(root, 1, (64, 64, 32), RES, (64, 64, 32))
    name = "chunk_0_0_0.0.1X"
    fixtures.create_shard(f"{root}/data/{name}.data", f"{root}/meta/{name}.meta",
                          np.full((64, 64, 32), 5, dtype=np.uint16), (32, 32, 32))
    with open(f"{root}/.sisf_access", "w") as f:
        f.write(fixtures.TOKEN + "\n")
    with open(f"{root}/meta/{name}.meta", "r+b") as f:
        f.seek(86)
        sizes = [struct.unpack(fixtures.SHARD_LINE_LAYOUT, f.read(12))[1] for _ in range(4)]  # 2x2x1 chunks
        if max(sizes) >= 60:
            raise RuntimeError(f"{sid}: a frame is not shorter than the video header: {sizes}")
        f.seek(6)  # version, dtype, channels, compression
        f.write(struct.pack("<H", 2))
    server.snapshot(ds)
    b = box(0, 64, 0, 64, 0, 32)
    results[f"{sid}: viewer read"] = digest(*server.request("GET", f"/{ds}/1/{b}"))
    results[f"{sid}: portal read"] = digest(*server.request("GET", f"/{ds}+token={fixtures.TOKEN}/1/{b}"))
    results[f"{sid}: PATCH part of a chunk"] = digest(*server.request(
        "PATCH", f"/{ds}+token={fixtures.TOKEN}/write/1/" + box(8, 24, 8, 24, 0, 16),
        np.full(16 * 16 * 16, 3, dtype=np.uint16).tobytes()))


def _unloadable_dataset(server, results, sid, ds, damage):
    """A dataset whose metadata.bin cannot give an mchunk size. Its first
    request runs the inventory scan, which divides by the mchunk size; on
    production (amd64) that is SIGFPE, and every restart dies in the same
    scan while the file is there, so it is removed before the restart and
    after the request."""
    root = os.path.join(server.data_dir, ds)
    _vol(root)
    damage(os.path.join(root, "metadata.bin"))
    remove = lambda: shutil.rmtree(root, ignore_errors=True)
    results[f"{sid}: /info"] = digest(*server.request("GET", f"/{ds}/info", before_restart=remove))
    remove()


def sc_empty_metadata(server, results, sid):
    """A 0-byte metadata.bin, as between the converter's open('wb') and its
    write, or after a conversion killed there. Production's mchunk size is
    whatever the heap held, so it dies only when that is 0."""
    _unloadable_dataset(server, results, sid, "sc_meta_empty", lambda p: open(p, "wb").close())


def _zero_mchunk_x(path):
    with open(path, "r+b") as f:
        f.seek(6)  # version, dtype, channels, then mchunk x
        f.write(struct.pack("<H", 0))


def sc_zero_mchunk_size(server, results, sid):
    """A complete metadata.bin whose mchunk x size is 0."""
    _unloadable_dataset(server, results, sid, "sc_meta_zero", _zero_mchunk_x)


def _slice_check(full, part, n, channels):
    """Whether part is channel n of full, a read of every channel (each
    channel is one contiguous block, in channel order)."""
    size = len(full) // channels
    ok = len(full) % channels == 0 and part == full[n * size:(n + 1) * size]
    return {"status": "match" if ok else "mismatch", "len": None, "sha256": None,
            "text": f"{len(part)} of {len(full)} bytes"}


def sc_channel_filter(server, results, sid):
    """+channel=N returns channel N only. Each such read must equal channel
    N of the same read without the filter (a box, a plane, a neuroglancer
    chunk, a coarser level, a projection and a gaussian filter, which works
    per channel); a channel the dataset does not have, one that is not a
    whole number, or a channel item that cannot be parsed answers 400.
    Production ignores the filter and returns every channel."""
    cases = [
        ("box", "", 1, box(0, 70, 0, 60, 0, 20)),
        ("plane", "", 1, box(3, 67, 5, 55, 10, 11)),
        ("ng chunk", "", 1, box(32, 64, 32, 60, 0, 20)),
        ("level 2", "", 2, box(0, 35, 0, 30, 0, 10)),
        ("max projection", "project=6", 1, box(0, 70, 0, 60, 4, 5)),
        ("gaussian", "gaussian=1", 1, box(10, 30, 10, 30, 5, 10)),
    ]
    for tag, extra, scale, b in cases:
        st, full = server.request("GET", f"/vol3c{'+' + extra if extra else ''}/{scale}/{b}")
        results[f"{sid}: {tag}, every channel"] = digest(st, full)
        for n in range(3):
            filt = f"channel={n}" + (f"&{extra}" if extra else "")
            st_n, part = server.request("GET", f"/vol3c+{filt}/{scale}/{b}")
            results[f"{sid}: {tag}, channel {n}"] = (_slice_check(full, part, n, 3) if st == st_n == 200
                                                     else digest(st_n, part))
    # A single-channel dataset: channel 0 is the whole read, on production too
    st, full = server.request("GET", "/vol1c/1/" + box(0, 64, 0, 64, 0, 32))
    st0, part = server.request("GET", "/vol1c+channel=0/1/" + box(0, 64, 0, 64, 0, 32))
    results[f"{sid}: vol1c channel 0"] = _slice_check(full, part, 0, 1)
    for bad in ("3", "x", "-1", "1.0"):
        results[f"{sid}: vol3c channel {bad}"] = digest(*server.request(
            "GET", f"/vol3c+channel={bad}/1/" + box(0, 8, 0, 8, 0, 1)))
    results[f"{sid}: vol1c channel 1"] = digest(*server.request("GET", "/vol1c+channel=1/1/" + box(0, 8, 0, 8, 0, 1)))
    # Items parse_filter_list drops: no value, no '=', two '=', and a second
    # '+' in the id (which drops every item)
    for tag, item in (("empty value", "channel="), ("no value", "channel"), ("two values", "channel=1=2"),
                      ("second +", "channel=1+offset=0")):
        results[f"{sid}: vol3c {tag}"] = digest(*server.request("GET", f"/vol3c+{item}/1/" + box(0, 8, 0, 8, 0, 1)))


def sc_read_limit(server, results, sid):
    """MAX_READ_VOXELS caps the box one read covers, per channel. A second
    server on the same data starts with the limit set to 70x60x19 voxels.
    On the image route a read of that many voxels, and one plane less,
    answer as without a limit, on the 3-channel fixture and on the 1-channel
    one; one plane more answers 400 on both (a dataset's channel count does
    not change the box it may read), and so do a projection of one plane
    that reads all 20, and that box with one channel selected. tracing answers
    400 when the box it reads around its two points is over the limit and as
    without one under it; so does raw_access. A server whose limit is not a
    number logs that it ignores it and reads as without one. Production
    ignores the variable."""
    limit = 70 * 60 * 19
    for env, cases in (
            ({"MAX_READ_VOXELS": str(limit)}, [
                ("under the limit", "/vol3c/1/" + box(0, 70, 0, 60, 0, 18)),
                ("at the limit", "/vol3c/1/" + box(0, 70, 0, 60, 0, 19)),
                ("one plane over", "/vol3c/1/" + box(0, 70, 0, 60, 0, 20)),
                ("1 channel, at the limit", "/vol1c/1/" + box(0, 70, 0, 60, 0, 19)),
                ("1 channel, one plane over", "/vol1c/1/" + box(0, 70, 0, 60, 0, 20)),
                # project=N reads the box's plane and the N after it
                ("projection at the limit", "/vol3c+project=18/1/" + box(0, 70, 0, 60, 0, 1)),
                ("projection over", "/vol3c+project=20/1/" + box(0, 70, 0, 60, 0, 1)),
                ("reversed box", "/vol3c/1/" + box(8, 2, 0, 8, 0, 1)),
                ("whole box, one channel", "/vol3c+channel=1/1/" + box(0, 70, 0, 60, 0, 20)),
                # The box is the points' bounding box widened by 5 and clamped
                # to the image: 12x11x11 here, 70x60x20 below
                ("tracing under the limit", "/vol1c/tracing/10,10,10/11,10,10"),
                ("tracing over", "/vol3c/tracing/0,0,0/69,59,19"),
                # vol1c's mchunk (0,0,0) stores 64x64x32
                ("raw_access at the limit", "/vol1c/raw_access/0,0,0,0/1/" + box(0, 64, 0, 64, 0, 19)),
                ("raw_access over", "/vol1c/raw_access/0,0,0,0/1/" + box(0, 64, 0, 64, 0, 20))]),
            ({"MAX_READ_VOXELS": "lots"}, [("limit not a number", "/vol3c/1/" + box(0, 70, 0, 60, 0, 20))])):
        tag = env["MAX_READ_VOXELS"]
        limited = Server(f"{server.role}-limit-{tag}", server.image, server.platform, server.data_dir, env)
        try:
            limited.start()
            for name, path in cases:
                results[f"{sid}: {name}"] = digest(*limited.request("GET", path))
            if tag != str(limit):
                log = "\n".join(limited.logs_tail(200))
                results[f"{sid}: limit not a number, logged"] = {
                    "status": "logged" if "MAX_READ_VOXELS ignored" in log else "not logged",
                    "len": None, "sha256": None, "text": None}
        finally:
            limited.remove()


def sc_raw_access_outside(server, results, sid):
    """raw_access over a range wider than the mchunk's stored tile (vol1c's
    mchunk (0,0,0) is 64 px wide). Production reads past the chunk buffer."""
    results[f"{sid}: read"] = digest(*server.request("GET", "/vol1c/raw_access/0,0,0,0/1/" + box(0, 100, 0, 10, 0, 1)))


def _encoded_read(server, results, key, path, accept_encoding):
    """Two results for a GET sent with accept_encoding: key, the body after
    decoding it (it must equal production's, which never compresses), and
    key + ", headers", its Content-Encoding and Vary and that body's SHA-256
    (plus any Content-Length that is not the length sent, and a gzip body
    that is not shorter than what it decodes to)."""
    _record_encoded(results, key, *server.get_encoded(path, accept_encoding))


def _record_encoded(results, key, st, headers, raw, note=""):
    """Records one answer as _encoded_read describes; headers maps lower-case
    names to values, and note is added at the end of the headers record."""
    body, ce = raw, headers.get("content-encoding")
    if isinstance(raw, bytes) and ce is not None:
        try:
            body = gzip.decompress(raw) if ce == "gzip" else f"unknown Content-Encoding {ce}"
        except Exception as e:
            body = f"not gzip: {type(e).__name__}: {e}"
    results[key] = digest(st, body)
    text = f"Content-Encoding: {ce or '-'}; Vary: {headers.get('vary') or '-'}"
    if isinstance(body, bytes):
        text += f"; decoded sha256 {hashlib.sha256(body).hexdigest()}"
    if isinstance(raw, bytes):
        cl = headers.get("content-length")
        if cl is not None and cl != str(len(raw)):
            text += f"; Content-Length {cl} but {len(raw)} bytes sent"
        if ce == "gzip" and isinstance(body, bytes) and len(raw) >= len(body):
            text += "; not shorter than the body"
    results[key + ", headers"] = {"status": st, "len": None, "sha256": None, "text": text + note}


def _startup_line(log, prefix):
    for line in log:
        if line.startswith(prefix):
            return line
    return "none"


KA_THREADS, KA_PER_THREAD = 16, 28


def _first_values(pairs):
    out = {}
    for k, v in pairs:
        out.setdefault(k, v)
    return out


def _keepalive_case(server, results, key, reqs):
    """Sends reqs, (tag, path, accept_encoding[, extra headers]) each, on one
    connection and records each answer as _encoded_read does under
    key + ": " + tag, its headers record adding the number of Connection
    headers and whether the request went out on a new connection."""
    answers = server.get_keepalive([r[1:] for r in reqs])
    for (tag, *_), (st, pairs, raw, reused) in zip(reqs, answers):
        n = sum(1 for k, _ in pairs if k == "connection")
        note = f"; Connection headers: {n}" + ("" if reused else "; on a new connection")
        _record_encoded(results, f"{key}: {tag}", st, _first_values(pairs), raw, note)
    for tag, *_ in reqs[len(answers):]:
        results[f"{key}: {tag}"] = {"status": "not sent", "len": None, "sha256": None,
                                    "text": "an earlier request on the connection failed"}


def _keepalive_stress(server, results, key, plan):
    """KA_THREADS threads, each on its own connection, send KA_PER_THREAD
    GETs, thread t taking plan[(t + i) % len(plan)] as its i-th, so every
    pair of neighbours in plan follows each other on some connection. Every
    answer must have the status and decoded body of the same request on a
    fresh connection, a Content-Length equal to what was sent, and go out on
    the thread's first connection. Records key (the count of problems, or
    CRASH) and key + ", second Connection headers" (answers with more than
    one Connection header)."""
    refs = {}
    for path, accept in plan:
        st, headers, raw = server.get_encoded(path, accept)
        if isinstance(raw, bytes) and headers.get("content-encoding") == "gzip":
            raw = gzip.decompress(raw)
        refs[(path, accept)] = (st, hashlib.sha256(raw).hexdigest() if isinstance(raw, bytes) else raw)
    problems, counts, lock = [], {"sent": 0, "dups": 0}, threading.Lock()

    def worker(t):
        conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=120)
        first = None
        try:
            for i in range(KA_PER_THREAD):
                path, accept = plan[(t + i) % len(plan)]
                where = f"thread {t} request {i} ({path}, {accept})"
                try:
                    conn.putrequest("GET", path, skip_accept_encoding=True)
                    if accept is not None:
                        conn.putheader("Accept-Encoding", accept)
                    conn.endheaders()
                    first = first or conn.sock
                    reused = conn.sock is first
                    r = conn.getresponse()
                    pairs = [(k.lower(), v) for k, v in r.getheaders()]
                    raw = r.read()
                except Exception as e:
                    with lock:
                        problems.append(f"{where}: {type(e).__name__}: {e}")
                    return
                headers = _first_values(pairs)
                bad = []
                if not reused:
                    bad.append("on a new connection")
                if headers.get("content-length") != str(len(raw)):
                    bad.append(f"Content-Length {headers.get('content-length')} but {len(raw)} bytes")
                try:
                    body = gzip.decompress(raw) if headers.get("content-encoding") == "gzip" else raw
                except Exception as e:
                    body = f"not gzip: {type(e).__name__}"
                if (r.status, hashlib.sha256(body).hexdigest() if isinstance(body, bytes) else body) != refs[(path, accept)]:
                    bad.append("answer differs from the read on a fresh connection")
                with lock:
                    counts["sent"] += 1
                    counts["dups"] += int(sum(1 for k, _ in pairs if k == "connection") > 1)
                    problems.extend(f"{where}: {b}" for b in bad)
        finally:
            conn.close()

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(KA_THREADS)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    text = f"{counts['sent']} answers on {KA_THREADS} connections, {len(problems)} problems"
    if problems:
        text += f"; first: {sorted(problems)[0]}"
    alive, code = server.running()
    if not alive:
        tail = server.logs_tail()
        server.restart()
        results[key] = {"status": "CRASH", "len": None, "sha256": None,
                        "text": f"exit={code}; {' | '.join(tail[-4:])}; {text}"}
    else:
        results[key] = {"status": "ok" if not problems else "problems", "len": None, "sha256": None, "text": text}
    results[key + ", second Connection headers"] = {"status": "count", "len": None, "sha256": None,
                                                    "text": str(counts["dups"])}


def sc_keepalive(server, results, sid):
    """Reads sent one after another on one HTTP/1.1 connection, each after
    the answer before it was read in full, as the portal's httpx pool and
    browsers send them. crow sends a body of 1 MiB or more synchronously
    inside res.end() and clears the response; production's crow then set a
    `connection: Keep-Alive` header on that cleared response, so the header
    stayed for the connection's next request: it went out again as a second
    Connection header, and that request freed it (set_header replaces it)
    while the rest of that answer's headers were still queued. asio sends 16
    buffers at a time and crow uses 4 per header, so an answer with enough
    headers of its own (the mesh route's Content-Type and
    Content-Disposition) then sends freed memory. The sanitizer build aborts
    there; a release build hands the freed block straight back for the same
    header, so only the second Connection header shows.
    Each case on its own connection: an image read over 1 MiB, then a mesh
    file, and the reverse; then GETs carrying Expect: 100-continue mixed with
    large and small reads, the last one with Connection: close. crow answers
    100 Continue and, since a GET has no body, queues the real answer behind
    it at once; production's crow finished the 100's write with the
    completion of a whole response, clearing res, the body copy (the first
    body byte went out as 0) and, for a close, the connection while the real
    answer's buffers were still queued. Each answer is recorded as
    _encoded_read does, its headers record adding the number of Connection
    headers. Then a short stress (see _keepalive_stress): bodies over 1 MiB
    and small ones, each followed on the same connection by another."""
    img, lab = "sc_ka_image", "sc_ka_labels"
    fixtures.untiled(os.path.join(server.data_dir, img), fixtures.pattern((1, 128, 128, 64), 17), (64, 64, 32),
                     RES, 1)
    lab_root = os.path.join(server.data_dir, lab)
    fixtures.segmentation(lab_root, (128, 128, 64), (64, 64, 32), RES, prefill_seed=17)
    os.makedirs(os.path.join(lab_root, "mesh"))
    with open(os.path.join(lab_root, "mesh", "1:0:0"), "wb") as f:
        f.write(np.arange(1000, dtype=np.float32).tobytes())
    big_img = f"/{img}/1/" + box(0, 128, 0, 128, 0, 64)  # 2 MiB
    big_lab = f"/{lab}/1/" + box(0, 128, 0, 128, 0, 64)  # 2 MiB
    chunk = f"/{lab}/1/" + box(32, 64, 32, 64, 0, 32)  # first voxel is not 0
    mesh = f"/{lab}/mesh/mesh/1:0:0"
    portal = f"/{lab}+token={fixtures.TOKEN}/1/" + box(0, 64, 0, 64, 0, 32)
    httpx = "gzip, deflate"
    _keepalive_case(server, results, f"{sid}: image over 1 MiB, then a mesh file",
                    [("1 image", big_img, httpx), ("2 mesh file", mesh, httpx)])
    _keepalive_case(server, results, f"{sid}: a mesh file, then an image over 1 MiB",
                    [("1 mesh file", mesh, httpx), ("2 image", big_img, httpx)])
    expect = (("Expect", "100-continue"),)
    _keepalive_case(server, results, f"{sid}: Expect: 100-continue among large and small reads",
                    [("1 image", big_img, httpx),
                     ("2 mesh file, Expect", mesh, httpx, expect),
                     ("3 labels over 1 MiB, Expect", big_lab, httpx, expect),
                     ("4 image, Expect", big_img, httpx, expect),
                     ("5 portal read", portal, httpx),
                     ("6 32^3 chunk, Expect", chunk, "gzip", expect),
                     ("7 portal read, Expect, Connection: close", portal, httpx,
                      expect + (("Connection", "close"),))])
    plan = [(big_img, httpx), (portal, httpx), (big_lab, None), (mesh, httpx), (chunk, "gzip"), (big_lab, httpx),
            (f"/{img}/1/" + box(0, 128, 0, 128, 10, 11), None)]
    _keepalive_stress(server, results, f"{sid}: stress", plan)


SWC_TABLES = """CREATE TABLE SWC(I INT NOT NULL, NEURONID INT NOT NULL, PARENTID INT NOT NULL, X REAL NOT NULL,
    Y REAL NOT NULL, Z REAL NOT NULL, R REAL NOT NULL, T INT NOT NULL, USERID INT NOT NULL,
    TIMESTAMP DATETIME DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE NEURONS(SOMAX REAL, SOMAY REAL, SOMAZ REAL, CELLTYPE INT, NOTES TEXT,
    TIMESTAMP DATETIME DEFAULT CURRENT_TIMESTAMP);"""


def _traces_dataset(root):
    """A small image dataset holding a traces.sql (the tables the CDN
    creates) with neurons 1 and 2 of two points each."""
    fixtures.untiled(root, fixtures.pattern((1, 64, 64, 32), 18), (64, 64, 32), RES, 1)
    con = sqlite3.connect(os.path.join(root, "traces.sql"))
    con.executescript(SWC_TABLES)
    for n in (1, 2):
        con.execute("INSERT INTO NEURONS (TIMESTAMP) VALUES ('2025-09-29 00:00:00')")
        con.execute("INSERT INTO SWC VALUES (1, ?, -1, ?, 20.5, 30.25, 1.5, 1, -1, '2025-09-29 00:00:00')", (n, 10.0 * n))
        con.execute("INSERT INTO SWC VALUES (2, ?, 1, ?, 21.5, 31.25, 1.5, 3, -1, '2025-09-29 00:00:00')", (n, 10.0 * n + 1))
    con.commit()
    con.close()


def _wait_visible(server, rel):
    """Waits until the server's container reads data_dir/rel with the bytes
    written on this side. A traces.sql read right after it was written here
    once answered "no such table" in the container (1 run in 3, after s17's
    stress), so the scenario does not send a request before this holds."""
    want = file_sha(os.path.join(server.data_dir, rel))
    for _ in range(60):
        out = subprocess.run(["docker", "exec", server.name, "sha256sum", f"/data/{rel}"],
                             capture_output=True, text=True).stdout.split()
        if out and out[0] == want:
            return
        time.sleep(0.5)
    raise RuntimeError(f"{server.role} does not see {rel} as written")


def _post_multipart(server, path, swc, timeout=120):
    """POSTs swc as the multipart part named "data", as nTracer uploads a
    neuron. Returns (status, body) as request() does."""
    b = "regressboundary7"
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"data\"; filename=\"n.swc\"\r\n"
            f"Content-Type: text/plain\r\n\r\n{swc}\r\n--{b}--\r\n").encode()
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=timeout)
    try:
        conn.request("POST", path, body=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
        r = conn.getresponse()
        return r.status, r.read()
    except Exception as e:
        return server._failed(e, timeout, None)
    finally:
        conn.close()


def _skeleton_sequence(server, results, key, ds):
    """ls, get, upload, replace, delete, then ls and get again, on ds."""
    swc = "1 1 5.5 6.5 7.5 1.0 -1\n2 3 5.75 6.75 7.75 1.0 1\n3 3 6.0 7.0 8.0 1.0 2\n"
    api = f"/{ds}/skeleton_api"
    results[f"{key}: ls"] = digest(*server.request("GET", f"{api}/ls"))
    results[f"{key}: get 1"] = digest(*server.request("GET", f"{api}/get/1"))
    results[f"{key}: upload"] = digest(*_post_multipart(server, f"{api}/upload", swc))
    results[f"{key}: replace 1"] = digest(*_post_multipart(server, f"{api}/replace/1", swc))
    results[f"{key}: delete 2 (a GET)"] = digest(*server.request("GET", f"{api}/delete/2"))
    results[f"{key}: ls after"] = digest(*server.request("GET", f"{api}/ls"))
    results[f"{key}: get 1 after"] = digest(*server.request("GET", f"{api}/get/1"))


def sc_skeleton_api_writes(server, results, sid):
    """skeleton_api's upload, replace and delete (a GET) change a dataset's
    traces.sql with no token. The fork refuses them with 403 unless the
    server runs with SKELETON_API_WRITES=1; ls and get still answer.
    Production has no such setting and always writes. Three servers: this
    one (unset), one with SKELETON_API_WRITES=1 on a dataset of its own
    (every answer as production's), and one with a value that is not 0 or 1
    (ignored with a log line, so writes stay off). traces.sql rows carry a
    TIMESTAMP set when written, so its bytes are not compared across
    servers; each server records whether the file changed instead."""
    off, on = "sc_skel", "sc_skel_on"
    for ds in (off, on):
        _traces_dataset(os.path.join(server.data_dir, ds))
    traces = os.path.join(server.data_dir, off, "traces.sql")
    built = file_sha(traces)
    for ds in (off, on):
        _wait_visible(server, f"{ds}/traces.sql")
    _skeleton_sequence(server, results, f"{sid}: unset", off)
    results[f"{sid}: unset: traces.sql"] = {"status": "unchanged" if file_sha(traces) == built else "changed",
                                            "len": None, "sha256": None, "text": None}
    results[f"{sid}: unset: startup line"] = {"status": "line", "len": None, "sha256": None,
                                              "text": _startup_line(server.logs_tail(10**6), "skeleton_api writes:")}
    for tag, env, ds in (("1", {"SKELETON_API_WRITES": "1"}, on), ("yes", {"SKELETON_API_WRITES": "yes"}, off)):
        other = Server(f"{server.role}-skeleton-{tag}", server.image, server.platform, server.data_dir, env)
        try:
            other.start()
            before = file_sha(os.path.join(server.data_dir, ds, "traces.sql"))
            if tag == "1":
                _skeleton_sequence(other, results, f"{sid}: SKELETON_API_WRITES={tag}", ds)
            else:
                results[f"{sid}: SKELETON_API_WRITES={tag}: delete 1 (a GET)"] = digest(
                    *other.request("GET", f"/{ds}/skeleton_api/delete/1"))
            changed = file_sha(os.path.join(server.data_dir, ds, "traces.sql")) != before
            results[f"{sid}: SKELETON_API_WRITES={tag}: traces.sql"] = {
                "status": "changed" if changed else "unchanged", "len": None, "sha256": None, "text": None}
            log = other.logs_tail(400)
            results[f"{sid}: SKELETON_API_WRITES={tag}: startup line"] = {
                "status": "line", "len": None, "sha256": None, "text": _startup_line(log, "skeleton_api writes:")}
            if tag == "yes":
                results[f"{sid}: SKELETON_API_WRITES={tag}: logged"] = {
                    "status": "logged" if any("SKELETON_API_WRITES ignored" in line for line in log) else "not logged",
                    "len": None, "sha256": None, "text": None}
        finally:
            other.stop()
            other.save_logs(os.path.join(os.path.dirname(server.data_dir), f"{other.role}.log"))
            other.remove()


def sc_replaced_during_read(server, results, sid):
    """An mchunk's .meta and .data replaced by rename (as mv, or rsync
    without --inplace, do), inside the container, while a long read of the
    mchunk is under way; then one voxel is read (which reloads the header),
    and after the long read ends the chunk it read last is read, one voxel
    of that chunk is PATCHed (so the rest of it is merged) and the chunk is
    read again. The fork's long read keeps reading the files it opened, so
    none of what it reads after the reload may reach the chunk cache.
    Production opens the files by name for every chunk. Every answer
    compared here is the same wherever the rename lands in the long read,
    for production and for a correct fork, so this case cannot fail
    spuriously; how likely it is to catch a fork that lets old chunks into
    the cache depends on the rename landing while the long read is still
    reading (4^3 chunks make it slow: 262,144 of them).
    stress_d6.py --mode rename measures that. The long read's own answer
    depends on where the rename lands and is compared by status only.
    Both versions are noise that zstd cannot compress, so every chunk is
    stored at the same length and the two .meta files are the same bytes:
    production, reading between the two renames, meets the new .data with
    the old table and reads a whole chunk of one version or the other,
    instead of a cut frame, which kills it (s2)."""
    ds, size, c = "sc_rename", (512, 512, 64), 4
    X, Y, Z = size
    name = "chunk_0_0_0.0.1X"
    root = os.path.join(server.data_dir, ds)
    fixtures.write_metadata(root, 1, size, RES, size)
    with open(os.path.join(root, ".sisf_access"), "w") as f:
        f.write(fixtures.TOKEN + "\n")
    old = np.random.default_rng(11).integers(0, 65536, size=size, dtype=np.uint16)
    fixtures.create_shard(f"{root}/data/{name}.data", f"{root}/meta/{name}.meta", old, (c, c, c))
    # The new version, outside any dataset on the same filesystem
    new = np.random.default_rng(12).integers(0, 65536, size=size, dtype=np.uint16)
    os.makedirs(os.path.join(server.data_dir, "sc_rename_next"))
    fixtures.create_shard(f"{server.data_dir}/sc_rename_next/{name}.data",
                          f"{server.data_dir}/sc_rename_next/{name}.meta", new, (c, c, c))
    one = f"/{ds}/1/" + box(0, 1, 0, 1, 0, 1)
    results[f"{sid}: first read"] = digest(*server.request("GET", one))

    long_read = {}
    th = threading.Thread(target=lambda: long_read.update(
        r=server.request("GET", f"/{ds}+project={Z}/1/" + box(0, X, 0, Y, 0, 1))))
    th.start()
    time.sleep(0.05)
    # In the container: a file shared from the host can be read with its old
    # contents for a moment after a rename on the host
    server.exec("sh", "-c", f"mv /data/sc_rename_next/{name}.data /data/{ds}/data/{name}.data && "
                            f"mv /data/sc_rename_next/{name}.meta /data/{ds}/meta/{name}.meta")
    results[f"{sid}: one voxel after the rename"] = digest(*server.request("GET", one))
    th.join()
    results[f"{sid}: long read"] = {"status": long_read["r"][0], "len": None, "sha256": None, "text": None}

    last = f"/{ds}/1/" + box(X - c, X, Y - c, Y, Z - c, Z)
    expect = np.ascontiguousarray(new[X - c:, Y - c:, Z - c:].transpose(2, 1, 0)).tobytes()
    st, body = server.request("GET", last)
    results[f"{sid}: chunk read last, read after"] = digest(st, body)
    results[f"{sid}: chunk read last, is the new files'"] = {
        "status": "match" if st == 200 and body == expect else "mismatch", "len": None, "sha256": None, "text": None}
    edit = f"/{ds}+token={fixtures.TOKEN}/write/1/" + box(X - c, X - c + 1, Y - c, Y - c + 1, Z - c, Z - c + 1)
    results[f"{sid}: PATCH one voxel of it"] = digest(*server.request("PATCH", edit, np.array([7], dtype=np.uint16).tobytes()))
    st, body = server.request("GET", last)
    results[f"{sid}: the chunk after the PATCH"] = digest(st, body)
    expect = np.frombuffer(expect, dtype=np.uint16).copy()
    expect[0] = 7
    results[f"{sid}: the chunk after the PATCH, is the new files' with the edit"] = {
        "status": "match" if st == 200 and body == expect.tobytes() else "mismatch",
        "len": None, "sha256": None, "text": None}


def sc_seg_gzip(server, results, sid):
    """SEG_GZIP=N sends a successful image read of a writable (protected)
    layer gzip-encoded (level N) to a client whose Accept-Encoding accepts
    gzip; nothing else is compressed. Second servers on the same data: with
    SEG_GZIP=1, =9, unset, =0 and a value that is not a level. Every body is
    compared with production's after decoding; its headers are recorded
    apart. With 1: reads of writable layers (a box, a 32^3 chunk, a plane,
    a projection, one channel, an empty segmentation, a body that stays
    over 1 MiB compressed (crow sends those in pieces), the portal's read
    with its token and httpx's header, a browser's header, a q value and
    upper case, a body of exactly 1024 bytes) are compressed; a body of
    1022 bytes or one voxel, labels that do not compress (noise), gzip
    refused with q=0 or with a q value that is not one, identity, deflate
    or "*" only, no Accept-Encoding at all, image layers, a 400, a 404,
    /info and raw_access are not. Unset, 0 and the value that is not a
    level compress nothing (that value is logged). Production ignores the
    variable and Accept-Encoding."""
    noise = os.path.join(server.data_dir, "sc_gzip_noise")
    fixtures.write_metadata(noise, 1, (64, 64, 32), RES, (64, 64, 32))
    name = "chunk_0_0_0.0.1X"
    fixtures.create_shard(f"{noise}/data/{name}.data", f"{noise}/meta/{name}.meta",
                          np.random.default_rng(16).integers(0, 65536, size=(64, 64, 32), dtype=np.uint16),
                          (32, 32, 32))
    with open(f"{noise}/.sisf_access", "w") as f:
        f.write(fixtures.TOKEN + "\n")
    # seg_prefilled as built: the write stage changes seg_prefilled itself
    fixtures.segmentation(os.path.join(server.data_dir, "sc_gzip_labels"), (100, 90, 37), (64, 64, 32), RES,
                          prefill_seed=5)
    fixtures.segmentation(os.path.join(server.data_dir, "sc_gzip_empty"), (128, 128, 64), (64, 64, 32), RES)
    fixtures.segmentation(os.path.join(server.data_dir, "sc_gzip_big"), (256, 256, 64), (128, 128, 64), RES,
                          prefill_seed=16)
    seg_box = "/sc_gzip_labels/1/" + box(0, 100, 0, 90, 0, 37)
    portal = f"/sc_gzip_labels+token={fixtures.TOKEN}/1/" + box(0, 64, 0, 64, 0, 32)
    on_cases = [
        ("writable layer, box", seg_box, "gzip"),
        ("writable layer, 32^3 chunk", "/sc_gzip_labels/1/" + box(32, 64, 32, 64, 0, 32), "gzip"),
        ("writable layer, plane", "/sc_gzip_labels/1/" + box(0, 100, 0, 90, 20, 21), "gzip"),
        ("writable layer, projection", "/sc_gzip_labels+project=5/1/" + box(0, 100, 0, 90, 3, 4), "gzip"),
        ("writable layer, channel 0", "/sc_gzip_labels+channel=0/1/" + box(0, 64, 0, 64, 0, 32), "gzip"),
        ("empty segmentation", "/sc_gzip_empty/1/" + box(0, 128, 0, 128, 0, 64), "gzip"),
        ("over 1 MiB compressed", "/sc_gzip_big/1/" + box(0, 256, 0, 256, 0, 64), "gzip"),
        ("portal read (token, httpx's header)", portal, "gzip, deflate"),
        ("browser's header", seg_box, "gzip, deflate, br, zstd"),
        ("q value and upper case", seg_box, "deflate;q=1, GZIP ; q=0.5"),
        ("1024 bytes", "/sc_gzip_labels/1/" + box(0, 32, 0, 16, 0, 1), "gzip"),
        ("1022 bytes", "/sc_gzip_labels/1/" + box(0, 73, 0, 7, 0, 1), "gzip"),
        ("one voxel", "/sc_gzip_labels/1/" + box(0, 1, 0, 1, 0, 1), "gzip"),
        ("labels that do not compress", "/sc_gzip_noise/1/" + box(0, 64, 0, 64, 0, 32), "gzip"),
        ("gzip refused (q=0)", seg_box, "gzip;q=0, deflate"),
        ("gzip with a q value that is not one", seg_box, "gzip;q=2"),
        ("identity only", seg_box, "identity"),
        ("deflate only", seg_box, "deflate"),
        ("* only", seg_box, "*"),
        ("no Accept-Encoding", seg_box, None),
        ("image layer", "/vol1c/1/" + box(0, 64, 0, 64, 0, 32), "gzip"),
        ("image layer, 3 channels", "/vol3c/1/" + box(0, 70, 0, 60, 0, 20), "gzip"),
        ("writable layer, box out of range (400)", "/sc_gzip_labels/1/" + box(0, 200, 0, 8, 0, 1), "gzip"),
        ("unknown dataset (404)", "/nope/1/" + box(0, 8, 0, 8, 0, 1), "gzip"),
        ("writable layer, /info", "/sc_gzip_labels/info", "gzip"),
        ("writable layer, raw_access", "/sc_gzip_labels/raw_access/0,0,0,0/1/" + box(0, 64, 0, 64, 0, 32), "gzip"),
    ]
    few_cases = [("writable layer, box", seg_box, "gzip"),
                 ("portal read (token, httpx's header)", portal, "gzip, deflate")]
    for tag, cases in (("1", on_cases), ("9", few_cases), ("unset", few_cases), ("0", few_cases),
                       ("fast", few_cases)):
        env = {} if tag == "unset" else {"SEG_GZIP": tag}
        other = Server(f"{server.role}-gzip-{tag}", server.image, server.platform, server.data_dir, env)
        try:
            other.start()
            for name_, path, accept in cases:
                _encoded_read(other, results, f"{sid}: SEG_GZIP={tag}: {name_}", path, accept)
            log = other.logs_tail(200)
            results[f"{sid}: SEG_GZIP={tag}: startup line"] = {
                "status": "line", "len": None, "sha256": None,
                "text": _startup_line(log, "Gzip for writable layers:")}
            if tag == "fast":
                results[f"{sid}: SEG_GZIP={tag}: logged"] = {
                    "status": "logged" if any("SEG_GZIP ignored" in line for line in log) else "not logged",
                    "len": None, "sha256": None, "text": None}
        finally:
            # Kept beside <work>/<role>.log: these are the only servers that
            # compress, so a sanitizer build's report for that code is here
            other.stop()
            other.save_logs(os.path.join(os.path.dirname(server.data_dir), f"{other.role}.log"))
            other.remove()


# Cases where production dies, hangs or loses data. Each builds its own dataset
# while the server runs (the first request for it triggers the inventory re-scan).
# s7 runs last: production dies in it, and nothing should depend on a server
# that has just been through it.
SCENARIOS = [
    ("s1 stale geometry", sc_stale_geometry),
    ("s2 corrupt zstd frame", sc_corrupt_zstd),
    ("s3 bad compression type", sc_bad_compression),
    ("s4 missing .data", sc_missing_data),
    ("s5 unreadable chunk under a PATCH", sc_unreadable_chunk_write),
    ("s6 read failure before a PATCH", sc_read_failure_before_patch),
    ("s9 cold .meta before a PATCH", sc_cold_meta_before_patch),
    ("s10 short video frame", sc_short_video_frame),
    ("s11 empty metadata.bin", sc_empty_metadata),
    ("s12 zero mchunk size", sc_zero_mchunk_size),
    ("s17 keep-alive connections", sc_keepalive),
    ("s18 skeleton_api writes", sc_skeleton_api_writes),
    ("s13 channel filter", sc_channel_filter),
    ("s14 read limit", sc_read_limit),
    ("s15 files replaced during a read", sc_replaced_during_read),
    ("s16 gzip for writable layers", sc_seg_gzip),
    ("s8 raw_access outside the mchunk", sc_raw_access_outside),
    ("s7 tile regrown in place", sc_regrown_tile),
]
SCENARIO_DATASETS = ["sc_stale", "sc_zstd", "sc_comp", "sc_nodata", "sc_short", "sc_strict", "sc_regrow", "sc_regrow_w",
                     "sc_cold", "sc_video", "sc_ka_image", "sc_ka_labels", "sc_rename", "sc_gzip_labels",
                     "sc_gzip_noise", "sc_gzip_empty", "sc_gzip_big"]


def run_scenarios(server, results):
    for sid, fn in SCENARIOS:
        fn(server, results, sid)
        alive, code = server.running()
        results[f"{sid} (alive after)"] = {"status": "alive" if alive else f"dead exit={code}",
                                           "len": None, "sha256": None, "text": None}
        if not alive:
            server.restart()


def reread(server, info_by_ds, results, tag):
    for ds in ("seg_empty", "seg_prefilled", "seg_holes"):
        info = info_by_ds[ds]
        for sc in info["scales"]:
            X, Y, Z = sc["size"]
            p = f"/{ds}/{sc['key']}/" + box(0, X, 0, Y, 0, Z)
            results[f"{tag} GET {p}"] = digest(*server.request("GET", p))


def disk_hashes(root):
    out = {}
    for ds in ["seg_empty", "seg_prefilled", "seg_holes"] + SCENARIO_DATASETS:
        for dirpath, _, files in os.walk(os.path.join(root, ds)):
            for f in files:
                path = os.path.join(dirpath, f)
                out[os.path.relpath(path, root)] = file_sha(path)
    return out


def phase(servers, fn):
    """Run fn(server, results) on both servers at once; return {role: results}."""
    res = {s.role: {} for s in servers}
    threads = [threading.Thread(target=fn, args=(s, res[s.role])) for s in servers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return res


def pin_matches(want, got):
    """want holds any of status, len, sha256, text (exact) and text_prefix."""
    if not isinstance(want, dict):
        return False
    for key, val in want.items():
        if key == "text_prefix":
            if not (got.get("text") or "").startswith(val):
                return False
        elif got.get(key) != val:
            return False
    return True


def unpinned(want, got):
    """An expected answer that accepts too much: none at all, or one that
    does not pin the body of an answer that has one (a status, or a status
    and a length, without sha256, text or text_prefix)."""
    if not want:
        return True
    if not isinstance(want, dict):
        return False
    body_pins = {"sha256", "text", "text_prefix"}
    has_body = bool(got.get("len"))
    return has_body and not (set(want) & body_pins)


def compare(a, b, allow):
    diffs = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        # Two crashes are the same behaviour even though their log tails differ.
        both_crashed = x is not None and y is not None and x["status"] == y["status"] == "CRASH"
        same = both_crashed or (x is not None and y is not None and x["status"] == y["status"]
                                and x["sha256"] == y["sha256"] and (x["sha256"] is not None or x["text"] == y["text"]))
        if not same:
            diffs.append({"id": k, "baseline": x, "candidate": y, "allowed": allow[k]["reason"] if k in allow else None})
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline-platform")
    ap.add_argument("--candidate-platform")
    ap.add_argument("--work", required=True)
    ap.add_argument("--allow")
    ap.add_argument("--report")
    ap.add_argument("--skip-danger", action="store_true")
    ap.add_argument("--candidate-env", action="append", default=[], metavar="KEY=VALUE",
                    help="environment variable for the candidate server, e.g. CHUNK_CACHE_LINES=4096; repeatable")
    args = ap.parse_args()
    candidate_env = {}
    for kv in args.candidate_env:
        key, sep, value = kv.partition("=")
        if not sep or not key:
            ap.error(f"--candidate-env takes KEY=VALUE, not {kv!r}")
        candidate_env[key] = value

    allow = {}
    if args.allow:
        with open(args.allow) as f:
            allow = {e["id"]: e for e in json.load(f)}

    work = os.path.abspath(args.work)
    shutil.rmtree(work, ignore_errors=True)
    fixtures.main(os.path.join(work, "fixtures"))
    servers = []
    for role, image, plat, env in (("baseline", args.baseline, args.baseline_platform, {}),
                                   ("candidate", args.candidate, args.candidate_platform, candidate_env)):
        d = os.path.join(work, role)
        shutil.copytree(os.path.join(work, "fixtures"), d)
        servers.append(Server(role, image, plat, d, env))

    report = {"baseline": args.baseline, "candidate": args.candidate, "candidate_env": candidate_env, "diffs": [],
              "counts": {}, "results": {}, "crashes": [], "timeouts": []}
    try:
        for s in servers:
            s.start()
        base = servers[0]
        info_by_ds = {}
        for ds in READ_DATASETS:
            st, body = base.request("GET", f"/{ds}/info")
            info_by_ds[ds] = json.loads(body) if st == 200 else None
        missing = [ds for ds, i in info_by_ds.items() if i is None]
        if missing:
            raise RuntimeError(f"baseline did not serve /info for {missing}; fixtures or mount are wrong")
        corpus = read_corpus(info_by_ds)

        def reads(server, results):
            for rid, method, path in corpus:
                results[rid] = digest(*server.request(method, path))

        stages = [("reads", reads), ("writes", run_writes),
                  ("reread", lambda s, r: reread(s, info_by_ds, r, "after-writes"))]
        if not args.skip_danger:
            stages += [("danger", run_danger), ("reread2", lambda s, r: reread(s, info_by_ds, r, "after-danger")),
                       ("scenarios", run_scenarios)]
        seg_full = "GET /seg_empty/1/" + box(0, 160, 0, 150, 0, 70) + " [full]"
        for name, fn in stages:
            t0 = time.time()
            res = phase(servers, fn)
            d = compare(res["baseline"], res["candidate"], allow)
            report["counts"][name] = {"requests": len(res["baseline"]), "diffs": len(d),
                                      "seconds": round(time.time() - t0, 1)}
            report["diffs"] += [dict(x, stage=name) for x in d]
            report["results"].update({f"{name}: {k}": v for k, v in res["baseline"].items()})
            for role in ("baseline", "candidate"):
                report["crashes"] += [{"role": role, "stage": name, "id": k, "text": v["text"]}
                                      for k, v in res[role].items() if v["status"] == "CRASH"]
                report["timeouts"] += [{"role": role, "stage": name, "id": k, "text": v["text"]}
                                       for k, v in res[role].items() if v["status"] == "TIMEOUT"]
            print(f"{name}: {len(res['baseline'])} requests, {len(d)} differ", flush=True)
            if name == "reads":
                before = res["baseline"].get(seg_full)
            if name == "reread":
                after = res["baseline"].get("after-writes GET /seg_empty/1/" + box(0, 160, 0, 150, 0, 70))
                # A write stage that changed nothing would make every write
                # comparison pass trivially; refuse to report that as coverage.
                if not before or not after or before["status"] != 200 or before["sha256"] == after["sha256"]:
                    raise RuntimeError("write stage did not change seg_empty on the baseline; writes were not exercised")
    finally:
        for s in servers:
            s.stop()
            s.save_logs(os.path.join(work, f"{s.role}.log"))

    ha, hb = disk_hashes(servers[0].data_dir), disk_hashes(servers[1].data_dir)
    disk = [{"id": f"disk {k}", "baseline": {"status": "file", "sha256": ha.get(k), "len": None, "text": None},
             "candidate": {"status": "file", "sha256": hb.get(k), "len": None, "text": None},
             "allowed": allow[f"disk {k}"]["reason"] if f"disk {k}" in allow else None, "stage": "disk"}
            for k in sorted(set(ha) | set(hb)) if ha.get(k) != hb.get(k)]
    report["counts"]["disk"] = {"files": len(set(ha) | set(hb)), "diffs": len(disk)}
    report["diffs"] += disk
    print(f"disk: {len(set(ha) | set(hb))} files, {len(disk)} differ")
    for s in servers:
        s.remove()

    unexpected = [d for d in report["diffs"] if not d["allowed"]]
    report["unexpected"] = len(unexpected)
    candidate_failures = [c for c in report["crashes"] + report["timeouts"] if c["role"] == "candidate"]
    report["candidate_crashes_or_timeouts"] = len(candidate_failures)
    seen = {d["id"] for d in report["diffs"]}
    not_seen = sorted(k for k in allow if k not in seen)
    report["expected_not_seen"] = not_seen
    # A listed difference must also be the one intended: the entry names the
    # candidate's answer ("expect") and the candidate must give it.
    # "unchanged" means a file equal to the dataset as built. An entry whose
    # expect accepts too much (unpinned()) fails the run unless it says why
    # in "unpinned_reason". "expect_baseline", where present, pins the
    # baseline's answer the same way, e.g. to show the case reaches the crash.
    mismatched, not_pinned, unpinned_fail = [], [], []
    for d in report["diffs"]:
        entry = allow.get(d["id"])
        if entry is None:
            continue
        want, got = entry.get("expect"), d["candidate"] or {}
        bwant = entry.get("expect_baseline")
        if bwant is not None and not pin_matches(bwant, d["baseline"] or {}):
            mismatched.append({"id": d["id"], "role": "baseline", "expected": bwant, "got": d["baseline"]})
        if unpinned(want, got):
            (not_pinned if entry.get("unpinned_reason") else unpinned_fail).append(d["id"])
            continue
        if want == "unchanged":
            path = d["id"][len("disk "):]
            ref = servers[1].fixture_hashes.get(path)
            if ref is None and os.path.exists(os.path.join(work, "fixtures", path)):
                ref = file_sha(os.path.join(work, "fixtures", path))
            ok = d["stage"] == "disk" and ref is not None and got.get("sha256") == ref
        else:
            ok = pin_matches(want, got)
        if not ok:
            mismatched.append({"id": d["id"], "role": "candidate", "expected": want, "got": got})
    report["expected_value_mismatch"] = mismatched
    report["not_pinned"] = not_pinned
    report["unpinned"] = unpinned_fail
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)
    for d in report["diffs"]:
        mark = "allowed" if d["allowed"] else "UNEXPECTED"
        print(f"[{mark}] {d['stage']}: {d['id']}\n    baseline:  {d['baseline']}\n    candidate: {d['candidate']}")
    for c in report["crashes"]:
        print(f"[CRASH] {c['role']} {c['stage']}: {c['id']}\n    {c['text']}")
    for c in report["timeouts"]:
        print(f"[TIMEOUT] {c['role']} {c['stage']}: {c['id']}\n    {c['text']}")
    for k in not_seen:
        print(f"[EXPECTED, NOT SEEN] {k}\n    {allow[k]['reason']}")
    for m in mismatched:
        print(f"[WRONG {m['role'].upper()} ANSWER] {m['id']}\n    expected:  {m['expected']}\n    {m['role']}: {m['got']}")
    for k in not_pinned:
        print(f"[ALLOWED WITHOUT AN EXPECTED ANSWER] {k}\n    {allow[k]['unpinned_reason']}")
    for k in unpinned_fail:
        print(f"[UNPINNED] {k}\n    expect: {allow[k].get('expect')!r} (give the full answer, or say why not in unpinned_reason)")
    print(f"RESULT: {len(report['diffs'])} differences, {len(unexpected)} unexpected, "
          f"{len(report['crashes'])} crashes, {len(report['timeouts'])} timeouts "
          f"(baseline and candidate counted separately); candidate crashed or timed out {len(candidate_failures)} times; "
          f"{len(not_seen)} expected differences not seen; {len(mismatched)} answers not as expected; "
          f"{len(unpinned_fail)} unpinned")
    return 1 if unexpected or candidate_failures or not_seen or mismatched or unpinned_fail else 0


if __name__ == "__main__":
    sys.exit(main())
