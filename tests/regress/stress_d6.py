"""Stress tests for stale chunks in the global chunk cache (D6).

--mode patch (the default): a viewer's read that decodes a chunk while a
PATCH replaces it can put its decode of the old chunk into the global chunk
cache after the PATCH has dropped that chunk's line; every later read of the
chunk is then served the old voxels until the line is evicted or the chunk
is written again.

The race sits between two points inside the server (a reader's table read
and its cache insert), so a black-box client cannot make it happen on
demand. This mode makes it likely instead: reader processes read one chunk
without pause while a writer PATCHes it with new contents each time
(incompressible, so each PATCH holds the cache lock for a while), and after
each PATCH, and a pause for reads still in flight, the chunk is read twice
more. Either of those reads returning older contents is a stale re-insert.

--mode rename: a request keeps an mchunk's .meta and .data open while it
reads the mchunk's chunks, and a descriptor stays on the file it opened when
another file is renamed over it. If the files are replaced that way during a
long read, and another request then notices the new .meta and reloads the
header (dropping the mchunk's cache lines), the long read must not put the
chunks it goes on reading from the old files into the cache; a PATCH that
merged into such a chunk would write the old voxels into the new files.
Each cycle: a long read of the whole mchunk (a projection, so the answer is
small) starts; the two files are replaced by rename with the next version;
a one-voxel read reloads the header; after the long read ends, the chunk it
read last is read, one voxel of that chunk is PATCHed (so the rest of it is
merged), and the chunk is read again. Either read showing anything but the
new version is stale. A cycle counts as landed when the long read was still
running well after the reload (by --margin seconds), i.e. it read chunks
after it. Harness scenario s15 is one such cycle.

--mode rename-write: the same replacement while a long PATCH is under way:
one that covers every chunk of the mchunk except for a one-voxel border, so
it reads and merges every chunk on the box's faces before it writes. The
rename comes at a later point in each cycle (--delay, then --sweep spread
over the cycles), so it lands before, during and after the PATCH's reads and
writes. Afterwards the whole mchunk is read: outside the box only the new
version may show, and inside only the new version or the PATCH's value. A
PATCH that the replacement interrupted may answer 500.

The copies and renames run inside the server's container: a file shared into
a container from the host can be read with its old contents for a moment
after a rename on the host, which would look like the stale reads these
modes look for.

    python stress_d6.py --image IMAGE --work DIR [--platform linux/arm64]
                        [--mode patch] [--writes 300] [--readers 8]
    python stress_d6.py --image IMAGE --work DIR --mode rename [--cycles 20]
                        [--chunk 8] [--delay 0.05]
    python stress_d6.py --image IMAGE --work DIR --mode rename-write
                        [--cycles 12] [--delay 0.05] [--sweep 5.5]

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
import threading
import time

import numpy as np

import fixtures

DS = "d6_seg"
BOX = "0-32_0-32_0-32"  # chunk 0 of the only mchunk, covered whole, so a PATCH never reads it first
RES = (650, 650, 1500)
N = 32 * 32 * 32

RENAME_DS = "d6_rename"
RENAME_SIZE = (512, 512, 128)
MCHUNK_NAME = "chunk_0_0_0.0.1X"


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


def start_server(args, work):
    """Starts the image on work; returns (container name, port)."""
    name = f"cdn-stress-d6-{os.getpid()}"
    cmd = ["docker", "run", "-d", "--name", name, "-p", "127.0.0.1::6000", "-v", f"{work}:/data"]
    if args.platform:
        cmd += ["--platform", args.platform]
    subprocess.run(cmd + [args.image], check=True, capture_output=True)
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
    except BaseException:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        raise
    return name, port


def run_patch(args, work):
    fixtures.segmentation(os.path.join(work, DS), (64, 64, 32), (64, 64, 32), RES)
    name, port = start_server(args, work)
    stale, procs = [], []
    stop, count, errors = mp.Event(), mp.Value("q", 0), mp.Queue()
    try:
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


def request(port, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    c.request(method, path, body)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def answer(vol, x0, x1, y0, y1, z0, z1):
    """What a read of the box returns for vol, indexed [x, y, z] (x fastest)."""
    return np.ascontiguousarray(vol[x0:x1, y0:y1, z0:z1].transpose(2, 1, 0)).reshape(-1)


def in_container(name, script):
    """Runs a shell script inside the server's container; see the module
    docstring for why the files are copied and renamed there."""
    subprocess.run(["docker", "exec", name, "sh", "-c", script], check=True, capture_output=True)


def build_rename_dataset(args, work):
    """The mchunk at three versions (same header, other voxels) under
    /data/versions, version 0 live; returns (versions, live paths in the
    container)."""
    c = args.chunk
    out = os.path.join(work, RENAME_DS)
    fixtures.write_metadata(out, 1, RENAME_SIZE, RES, RENAME_SIZE)
    with open(f"{out}/.sisf_access", "w") as f:
        f.write(fixtures.TOKEN + "\n")
    # Three, so that what a stale read returns can never equal the version
    # just put in place
    versions = [fixtures.pattern(RENAME_SIZE, seed) for seed in (1, 2, 3)]
    # Outside any dataset, but on the same filesystem as the live files, so a
    # rename replaces them in one step
    os.makedirs(os.path.join(work, "versions"))
    os.makedirs(os.path.join(work, "stage"))
    for i, vol in enumerate(versions):
        fixtures.create_shard(f"{work}/versions/{i}.data", f"{work}/versions/{i}.meta", vol, (c, c, c))
    for ext in ("data", "meta"):
        shutil.copyfile(f"{work}/versions/0.{ext}", f"{out}/{ext}/{MCHUNK_NAME}.{ext}")
    return versions, {ext: f"/data/{RENAME_DS}/{ext}/{MCHUNK_NAME}.{ext}" for ext in ("data", "meta")}


def stage_version(name, v):
    in_container(name, f"cp /data/versions/{v}.data /data/stage/next.data && cp /data/versions/{v}.meta /data/stage/next.meta")


def replace_live(name, live):
    in_container(name, f"mv -f /data/stage/next.data {live['data']} && mv -f /data/stage/next.meta {live['meta']}")


def run_rename(args, work):
    X, Y, Z = RENAME_SIZE
    c = args.chunk
    versions, live = build_rename_dataset(args, work)

    last = (X - c, X, Y - c, Y, Z - c, Z)  # the chunk the long read reads last
    edit = (X - c, X - c + 1, Y - c, Y - c + 1, Z - c, Z - c + 1)
    edit_path = f"/{RENAME_DS}+token={fixtures.TOKEN}/write/1/" + "{}-{}_{}-{}_{}-{}".format(*edit)
    last_path = f"/{RENAME_DS}/1/" + "{}-{}_{}-{}_{}-{}".format(*last)
    long_path = f"/{RENAME_DS}+project={Z}/1/0-{X}_0-{Y}_0-1"

    name, port = start_server(args, work)
    rows = []
    try:
        st, _ = request(port, "GET", f"/{RENAME_DS}/1/0-1_0-1_0-1")  # builds the reader
        if st != 200:
            raise RuntimeError(f"first read answered {st}")
        for k in range(args.cycles):
            old, new = versions[k % 3], versions[(k + 1) % 3]
            stage_version(name, (k + 1) % 3)
            res = {}

            def long_read():
                res["R"] = request(port, "GET", long_path)
                res["R_done"] = time.time()

            th = threading.Thread(target=long_read)
            t0 = time.time()
            th.start()
            time.sleep(args.delay)
            replace_live(name, live)
            st2, _ = request(port, "GET", f"/{RENAME_DS}/1/0-1_0-1_0-1")
            t_reload = time.time()
            th.join()
            if res["R"][0] != 200 or st2 != 200:
                raise RuntimeError(f"cycle {k}: long read {res['R'][0]}, one-voxel read {st2}")
            landed = res["R_done"] - t_reload > args.margin

            def which(body, vol_new, vol_old, skip_first=False):
                a = np.frombuffer(body, dtype=np.uint16)
                n_, o_ = answer(vol_new, *last), answer(vol_old, *last)
                if skip_first:
                    a, n_, o_ = a[1:], n_[1:], o_[1:]
                return "new" if np.array_equal(a, n_) else ("old" if np.array_equal(a, o_) else "neither")

            st3, b3 = request(port, "GET", last_path)
            read_after = which(b3, new, old) if st3 == 200 else f"status {st3}"
            stp, _ = request(port, "PATCH", edit_path, np.array([7], dtype=np.uint16).tobytes())
            st4, b4 = request(port, "GET", last_path)
            edited = int(np.frombuffer(b4, dtype=np.uint16)[0]) if st4 == 200 else None
            written = which(b4, new, old, skip_first=True) if st4 == 200 else f"status {st4}"
            rows.append({"cycle": k, "landed": landed, "long_read_s": round(res["R_done"] - t0, 3),
                         "after_reload_s": round(res["R_done"] - t_reload, 3), "read_after": read_after,
                         "patch": stp, "edited_voxel": edited, "written": written})
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    landed = [r for r in rows if r["landed"]]
    stale_read = [r for r in rows if r["read_after"] != "new"]
    stale_write = [r for r in rows if r["written"] != "new" or r["patch"] != 200 or r["edited_voxel"] != 7]
    print(f"image {args.image}: {len(rows)} cycles of {args.chunk}^3 chunks, {len(landed)} landed "
          f"(long read still running {args.margin} s after the reload); "
          f"{len(stale_read)} cycles with a stale read, {len(stale_write)} with a stale or failed PATCH")
    for r in rows:
        print(f"  {r}")
    return 1 if stale_read or stale_write else 0


def run_rename_write(args, work):
    X, Y, Z = RENAME_SIZE
    versions, live = build_rename_dataset(args, work)

    # Every chunk on the box's faces is covered only in part, so the PATCH
    # reads and merges all of them before it writes anything
    b = (1, X - 1, 1, Y - 1, 1, Z - 1)
    inside = np.zeros(RENAME_SIZE, dtype=bool)
    inside[b[0]:b[1], b[2]:b[3], b[4]:b[5]] = True
    body = np.full((b[5] - b[4], b[3] - b[2], b[1] - b[0]), 9, dtype=np.uint16).tobytes()
    patch_path = f"/{RENAME_DS}+token={fixtures.TOKEN}/write/1/" + "{}-{}_{}-{}_{}-{}".format(*b)
    whole_path = f"/{RENAME_DS}/1/0-{X}_0-{Y}_0-{Z}"

    name, port = start_server(args, work)
    rows = []
    try:
        st, _ = request(port, "GET", f"/{RENAME_DS}/1/0-1_0-1_0-1")  # builds the reader
        if st != 200:
            raise RuntimeError(f"first read answered {st}")
        for k in range(args.cycles):
            new = versions[(k + 1) % 3]
            stage_version(name, (k + 1) % 3)
            delay = args.delay + args.sweep * k / max(1, args.cycles - 1)
            res = {}

            def patch():
                res["P"] = request(port, "PATCH", patch_path, body)
                res["P_done"] = time.time()

            th = threading.Thread(target=patch)
            t0 = time.time()
            th.start()
            time.sleep(delay)
            replace_live(name, live)
            t_ren = time.time()
            th.join()
            request(port, "GET", f"/{RENAME_DS}/1/0-1_0-1_0-1")  # reloads the header if nothing has
            st, whole = request(port, "GET", whole_path)
            if st != 200:
                raise RuntimeError(f"cycle {k}: whole read answered {st}")
            got = np.frombuffer(whole, dtype=np.uint16).reshape(Z, Y, X).transpose(2, 1, 0)
            # Outside the box only the new files' voxels may show; inside, those
            # or the PATCH's
            bad_out = int(((got != new) & ~inside).sum())
            bad_in = int(((got != new) & (got != 9) & inside).sum())
            painted = int(((got == 9) & inside & (new != 9)).sum())
            rows.append({"cycle": k, "rename_at_s": round(t_ren - t0, 3), "patch_s": round(res["P_done"] - t0, 3),
                         "patch": res["P"][0], "answer": res["P"][1][:80].decode("utf-8", "replace"),
                         "painted_in_new_files": painted,
                         "stale_outside": bad_out, "stale_inside": bad_in})
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    bad = [r for r in rows if r["stale_outside"] or r["stale_inside"] or r["patch"] not in (200, 500)]
    during = [r for r in rows if r["rename_at_s"] < r["patch_s"]]
    print(f"image {args.image}: {len(rows)} cycles of {args.chunk}^3 chunks, {len(during)} with the rename before "
          f"the PATCH answered ({sum(r['patch'] == 500 for r in rows)} PATCHes answered 500); "
          f"{len(bad)} cycles with old voxels written into the new files")
    for r in rows:
        print(f"  {r}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--platform")
    ap.add_argument("--work", required=True)
    ap.add_argument("--mode", choices=("patch", "rename", "rename-write"), default="patch")
    ap.add_argument("--writes", type=int, default=300, help="patch mode: PATCHes")
    ap.add_argument("--readers", type=int, default=8, help="patch mode: reader processes")
    ap.add_argument("--settle", type=float, default=0.03, help="patch mode: pause after each PATCH before checking, in s")
    ap.add_argument("--cycles", type=int, default=20, help="rename mode: replacements")
    ap.add_argument("--chunk", type=int, default=8, help="rename mode: chunk edge; small chunks make the long read long")
    ap.add_argument("--delay", type=float, default=0.05, help="rename mode: from the long read's start to the rename, in s")
    ap.add_argument("--sweep", type=float, default=2.0,
                    help="rename-write mode: the rename comes delay + sweep * k / (cycles - 1) s after the PATCH starts")
    ap.add_argument("--margin", type=float, default=0.2,
                    help="rename mode: how long the long read must go on after the reload for the cycle to count as landed, in s")
    args = ap.parse_args()

    work = os.path.abspath(args.work)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    return {"patch": run_patch, "rename": run_rename, "rename-write": run_rename_write}[args.mode](args, work)


if __name__ == "__main__":
    sys.exit(main())
