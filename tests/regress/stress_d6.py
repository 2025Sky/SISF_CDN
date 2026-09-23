"""Stress test for the stale cache re-insert (D6).

A viewer's read that decodes a chunk while a PATCH replaces it can put its
decode of the old chunk into the global chunk cache after the PATCH has
dropped that chunk's line; every later read of the chunk is then served the
old voxels until the line is evicted or the chunk is written again.

The race sits between two points inside the server (a reader's table read
and its cache insert), so a black-box client cannot make it happen on
demand. This script makes it likely instead: reader processes read one
chunk without pause while a writer PATCHes it with new contents each time
(incompressible, so each PATCH holds the cache lock for a while), and after
each PATCH, and a pause for reads still in flight, the chunk is read twice
more. Either of those reads returning older contents is a stale re-insert.

    python stress_d6.py --image IMAGE --work DIR [--platform linux/arm64]
                        [--writes 300] [--readers 8]

Exit status 1 if any stale read was seen. A run that sees none proves
nothing on its own: run the same settings against an image without the fix
first, and rely on them only if that run does see stale reads.
"""

import argparse
import http.client
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import fixtures

DS = "d6_seg"
BOX = "0-32_0-32_0-32"  # chunk 0 of the only mchunk, covered whole, so a PATCH never reads it first
RES = (650, 650, 1500)
N = 32 * 32 * 32


def contents(v):
    """Chunk contents for version v: noise, with v in the first voxel (the
    voxel at x, y, z = 0 comes first in both the request and the answer)."""
    a = np.random.default_rng(v).integers(0, 65536, size=N, dtype=np.uint16)
    a[0] = v
    return a


def get_version(conn):
    conn.request("GET", f"/{DS}/1/{BOX}")
    r = conn.getresponse()
    body = r.read()
    if r.status != 200 or len(body) != 2 * N:
        raise RuntimeError(f"read answered {r.status} with {len(body)} bytes")
    a = np.frombuffer(body, dtype=np.uint16)
    v = int(a[0])
    if v and not np.array_equal(a, contents(v)):
        raise RuntimeError(f"read returned version {v} with contents that are not that version's")
    return v


def reader(port, stop, count, errors):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    n = 0
    try:
        while not stop.is_set():
            conn.request("GET", f"/{DS}/1/{BOX}")
            r = conn.getresponse()
            r.read()
            n += 1
    except Exception as e:  # noqa: BLE001 - reported by the parent
        errors.put(str(e))
    with count.get_lock():
        count.value += n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--platform")
    ap.add_argument("--work", required=True)
    ap.add_argument("--writes", type=int, default=300)
    ap.add_argument("--readers", type=int, default=8)
    ap.add_argument("--settle", type=float, default=0.03, help="pause after each PATCH before checking, in s")
    args = ap.parse_args()

    work = os.path.abspath(args.work)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    fixtures.segmentation(os.path.join(work, DS), (64, 64, 32), (64, 64, 32), RES)

    name = f"cdn-stress-d6-{os.getpid()}"
    cmd = ["docker", "run", "-d", "--name", name, "-p", "127.0.0.1::6000", "-v", f"{work}:/data"]
    if args.platform:
        cmd += ["--platform", args.platform]
    subprocess.run(cmd + [args.image], check=True, capture_output=True)
    stale, procs = [], []
    stop, count, errors = mp.Event(), mp.Value("q", 0), mp.Queue()
    try:
        out = subprocess.run(["docker", "port", name, "6000/tcp"], check=True, capture_output=True, text=True).stdout
        port = int(out.strip().splitlines()[0].rsplit(":", 1)[1])
        deadline = time.time() + 120
        while True:
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                c.request("GET", "/version")
                if c.getresponse().status == 200:
                    break
            except Exception:
                if time.time() > deadline:
                    raise
                time.sleep(0.5)

        procs = [mp.Process(target=reader, args=(port, stop, count, errors)) for _ in range(args.readers)]
        for p in procs:
            p.start()

        wconn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        cconn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        t0 = time.time()
        for n in range(args.writes):
            v = (n % 60000) + 1
            wconn.request("PATCH", f"/{DS}+token={fixtures.TOKEN}/write/1/{BOX}", contents(v).tobytes())
            r = wconn.getresponse()
            r.read()
            if r.status != 200:
                raise RuntimeError(f"PATCH {n} answered {r.status}")
            time.sleep(args.settle)
            got = [get_version(cconn), get_version(cconn)]
            if any(g != v for g in got):
                stale.append((n, v, got))
            if not errors.empty():
                break
        elapsed = time.time() - t0
    finally:
        stop.set()
        for p in procs:
            p.join(timeout=30)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    errs = []
    while not errors.empty():
        errs.append(errors.get())
    print(f"image {args.image}: {args.writes} PATCHes in {elapsed:.1f} s, {count.value} concurrent reads "
          f"from {args.readers} processes, {len(stale)} PATCHes followed by a stale read")
    for n, v, got in stale[:10]:
        print(f"  PATCH {n} wrote version {v}; the two reads after it returned {got}")
    for e in errs:
        print(f"  ERROR {e}")
    return 1 if stale or errs else 0


if __name__ == "__main__":
    sys.exit(main())
