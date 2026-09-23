"""Black-box regression harness: run two CDN images on identical data, send
both the same requests, and report every response that differs.

    python harness.py --baseline IMAGE --candidate IMAGE --work DIR
                      [--baseline-platform linux/amd64] [--candidate-platform ...]
                      [--allow expected_diffs.json] [--report report.json]

Reads are compared by status code and SHA-256 of the body. Writes are sent to
each server's own copy of the data in the same order the SAVAII portal sends
them (read a 128-aligned box, merge labels, PATCH it back), then compared both
through re-reads and byte-for-byte on disk. A server that dies is recorded as
CRASH for that request, restarted, and the run continues.

Exit status: 0 when every difference is listed in the allow file and neither
server crashed differently from the other, 1 otherwise.
"""

import argparse
import hashlib
import json
import os
import shutil
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


class Server:
    def __init__(self, role, image, platform, data_dir):
        self.role, self.image, self.platform, self.data_dir = role, image, platform, data_dir
        self.name = f"cdn-regress-{role}-{os.getpid()}"
        self.port = None

    def start(self):
        cmd = ["docker", "run", "-d", "--name", self.name, "-p", "127.0.0.1::6000",
               "-v", f"{self.data_dir}:/data"]
        if self.platform:
            cmd += ["--platform", self.platform]
        subprocess.run(cmd + [self.image], check=True, capture_output=True)
        self._wait()

    def restart(self):
        subprocess.run(["docker", "start", self.name], check=True, capture_output=True)
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

    def stop(self):
        subprocess.run(["docker", "stop", "-t", "5", self.name], capture_output=True)

    def remove(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)

    def request(self, method, path, body=None):
        """Returns (status, body bytes). status is 'CRASH' if the server died,
        'NOCONN' if it is alive but the connection failed."""
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=body, method=method)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except Exception as e:  # connection reset, refused, timeout
            time.sleep(1)
            alive, code = self.running()
            if not alive:
                tail = self.logs_tail()
                self.restart()
                return "CRASH", f"exit={code}; {' | '.join(tail[-4:])}; {type(e).__name__}"
            return "NOCONN", f"{type(e).__name__}: {e}"


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
    # Straddles present mchunk (0,0,0) and the missing (1,0,0). Production skips
    # the missing part and answers 200; upstream without Bin's bcb9bb1
    # dereferences a null reader.
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


def reread(server, info_by_ds, results, tag):
    for ds in ("seg_empty", "seg_prefilled", "seg_holes"):
        info = info_by_ds[ds]
        for sc in info["scales"]:
            X, Y, Z = sc["size"]
            p = f"/{ds}/{sc['key']}/" + box(0, X, 0, Y, 0, Z)
            results[f"{tag} GET {p}"] = digest(*server.request("GET", p))


def disk_hashes(root):
    out = {}
    for ds in ("seg_empty", "seg_prefilled", "seg_holes"):
        for dirpath, _, files in os.walk(os.path.join(root, ds)):
            for f in files:
                path = os.path.join(dirpath, f)
                with open(path, "rb") as fh:
                    out[os.path.relpath(path, root)] = hashlib.sha256(fh.read()).hexdigest()
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


def compare(a, b, allow):
    diffs = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        # Two crashes are the same behaviour even though their log tails differ.
        both_crashed = x is not None and y is not None and x["status"] == y["status"] == "CRASH"
        same = both_crashed or (x is not None and y is not None and x["status"] == y["status"]
                                and x["sha256"] == y["sha256"] and (x["sha256"] is not None or x["text"] == y["text"]))
        if not same:
            diffs.append({"id": k, "baseline": x, "candidate": y, "allowed": allow.get(k)})
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
    args = ap.parse_args()

    allow = {}
    if args.allow:
        with open(args.allow) as f:
            allow = {e["id"]: e["reason"] for e in json.load(f)}

    work = os.path.abspath(args.work)
    shutil.rmtree(work, ignore_errors=True)
    fixtures.main(os.path.join(work, "fixtures"))
    servers = []
    for role, image, plat in (("baseline", args.baseline, args.baseline_platform),
                              ("candidate", args.candidate, args.candidate_platform)):
        d = os.path.join(work, role)
        shutil.copytree(os.path.join(work, "fixtures"), d)
        servers.append(Server(role, image, plat, d))

    report = {"baseline": args.baseline, "candidate": args.candidate, "diffs": [], "counts": {}, "results": {}, "crashes": []}
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
            stages += [("danger", run_danger), ("reread2", lambda s, r: reread(s, info_by_ds, r, "after-danger"))]
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

    ha, hb = disk_hashes(servers[0].data_dir), disk_hashes(servers[1].data_dir)
    disk = [{"id": f"disk {k}", "baseline": {"status": "file", "sha256": ha.get(k), "len": None, "text": None},
             "candidate": {"status": "file", "sha256": hb.get(k), "len": None, "text": None},
             "allowed": allow.get(f"disk {k}"), "stage": "disk"}
            for k in sorted(set(ha) | set(hb)) if ha.get(k) != hb.get(k)]
    report["counts"]["disk"] = {"files": len(set(ha) | set(hb)), "diffs": len(disk)}
    report["diffs"] += disk
    print(f"disk: {len(set(ha) | set(hb))} files, {len(disk)} differ")
    for s in servers:
        s.remove()

    unexpected = [d for d in report["diffs"] if not d["allowed"]]
    report["unexpected"] = len(unexpected)
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)
    for d in report["diffs"]:
        mark = "allowed" if d["allowed"] else "UNEXPECTED"
        print(f"[{mark}] {d['stage']}: {d['id']}\n    baseline:  {d['baseline']}\n    candidate: {d['candidate']}")
    for c in report["crashes"]:
        print(f"[CRASH] {c['role']} {c['stage']}: {c['id']}\n    {c['text']}")
    print(f"RESULT: {len(report['diffs'])} differences, {len(unexpected)} unexpected, "
          f"{len(report['crashes'])} crashes (baseline and candidate counted separately)")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
