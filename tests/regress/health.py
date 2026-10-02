"""Checks /health on one image, each run on fresh servers and data, and
prints every problem.

    python health.py --image IMAGE --work DIR [--platform linux/amd64]
                     [--lock-ms 700] [--runs 1] [--require-hook]

/health tries the chunk cache lock for HEALTH_LOCK_MS (default 2000 ms) and
answers 200 "ok lock_wait_ms=N" when it gets it, or 503 "stuck lock_wait_ms=N
write_held_ms=M" when it does not, M being how long the write holding the
lock has held it. Checked here:

- idle: 200, text/plain, "ok lock_wait_ms=N" with N under 100; ten probes
  leave /performance and every file under /data as they were; the same
  answer from a server started with READ_ONLY=1;
- HEALTH_LOCK_MS: 1 and 30000 are taken without a log line; "abc", "0" and
  "30001" are each ignored with one "HEALTH_LOCK_MS ignored" line and
  /health still answers 200;
- stuck, on one server with HEALTH_LOCK_MS unset (2000) and one with
  --lock-ms: while /data/.test_hold_write_lock exists, a PATCH waits holding
  the lock (only in a build made with -DNTRACER_TEST_HOOKS=ON). Three probes
  must each answer 503 within HEALTH_LOCK_MS + 0.5 s, with lock_wait_ms at
  least HEALTH_LOCK_MS and write_held_ms growing by the time between them;
  /version answers meanwhile. While a probe waits one more thread is in a
  futex wait, and once it has answered the count is back where it was, so a
  failed probe leaves no thread blocked. After the file is removed the PATCH
  answers 200 and reads back, /health answers 200 and the count is back to
  its idle value.

On a build without the hook the PATCH does not wait, and the held checks are
reported as SKIPPED; --require-hook makes that a failure. Threads are counted
from /proc/1/task/*/syscall inside the container (the server is PID 1),
using the futex syscall number of the kernel's architecture. Every request
opens its own connection, and fewer than 30 are sent while the write is
held: crow hands connections to its request threads in turn, so the 31st
would land on the held thread and wait like it.

Exit status 1 if any run found a problem.
"""

import argparse
import http.client
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

import fixtures
import harness

DS = "health_seg"
MARKER = "/data/.test_hold_write_lock"
HOOK_LINE = "TEST HOOK: write holding"
FUTEX = {"aarch64": "98", "x86_64": "202"}
OK = re.compile(r"ok lock_wait_ms=(\d+)\n")
STUCK = re.compile(r"stuck lock_wait_ms=(\d+) write_held_ms=(-?\d+)\n")
DEFAULT_LOCK_MS = 2000


def get(server, path, timeout, method="GET", body=None):
    """(status or 'TIMEOUT' / 'ERR:<name>', content type, body bytes, seconds)"""
    t0 = time.monotonic()
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=timeout)
    try:
        conn.request(method, path, body=body)
        r = conn.getresponse()
        data = r.read()
        return r.status, r.getheader("Content-Type"), data, time.monotonic() - t0
    except TimeoutError:
        return "TIMEOUT", None, b"", time.monotonic() - t0
    except Exception as e:  # connection reset, refused
        return f"ERR:{type(e).__name__}", None, b"", time.monotonic() - t0
    finally:
        conn.close()


def sh(server, script):
    return subprocess.run(["docker", "exec", server.name, "sh", "-c", script],
                          capture_output=True, text=True, check=True).stdout


class Threads:
    """Counts the server's threads that are in a futex wait."""

    def __init__(self, server, problems):
        self.server = server
        arch = sh(server, "cat /proc/sys/kernel/arch 2>/dev/null || uname -m").strip()
        self.futex = FUTEX.get(arch)
        if self.futex is None:
            problems.append(f"cannot count futex threads: no futex number for kernel architecture {arch!r}")

    def futex_count(self):
        if self.futex is None:
            return None
        calls = sh(self.server, "for t in /proc/1/task/*; do cut -d' ' -f1 $t/syscall; done").split()
        return calls.count(self.futex)

    def settle(self, want, tries=15):
        """Waits up to ~3 s for the count to be want; returns the last count."""
        n = None
        for _ in range(tries):
            n = self.futex_count()
            if n is None or n == want:
                return n
            time.sleep(0.2)
        return n


def files(server):
    return sh(server, "find /data -printf '%p %s %T@\\n' | sort")


def log_lines(server):
    r = subprocess.run(["docker", "logs", server.name], capture_output=True, text=True)
    return (r.stdout + r.stderr).splitlines()


def check_ok(server, tag, problems, timeout=5.0):
    st, ctype, data, dt = get(server, "/health", timeout)
    text = data.decode("utf-8", "replace")
    m = OK.fullmatch(text)
    if st != 200 or ctype != "text/plain" or m is None or int(m.group(1)) >= 100:
        problems.append(f"{tag}: /health answered {st} {ctype!r} {text!r} in {dt:.3f} s, "
                        "want 200 'text/plain' 'ok lock_wait_ms=N' with N < 100")
    return st, text


def idle_checks(server, problems):
    print(f"  idle: /health {check_ok(server, 'idle', problems)}")
    perf, before = get(server, "/performance", 5)[2], files(server)
    for i in range(10):
        check_ok(server, f"idle probe {i}", problems)
    if get(server, "/performance", 5)[2] != perf:
        problems.append("ten /health probes changed /performance")
    if files(server) != before:
        problems.append("ten /health probes changed a file under /data")
    return perf, before


def held_checks(server, tag, lock_ms, threads, problems):
    """Probes while the hook holds the lock."""
    before = threads.futex_count()
    version = get(server, "/version", 5)
    print(f"  {tag}: write held; /version {version[0]} {version[2]!r} in {version[3]:.3f} s")
    if version[0] != 200 or version[3] > 1:
        problems.append(f"{tag}: /version answered {version[0]} in {version[3]:.3f} s while the write is held")

    probes, during = [], []
    for i in range(3):
        res = []
        prober = threading.Thread(target=lambda: res.append(get(server, "/health", lock_ms / 1000 + 0.5)))
        prober.start()
        if i == 0 and before is not None:  # while it waits, one more thread is in a futex wait
            t_end = time.monotonic() + lock_ms / 1000 * 0.75
            time.sleep(lock_ms / 1000 * 0.25)
            while time.monotonic() < t_end and before + 1 not in during:
                during.append(threads.futex_count())
        prober.join()
        probes.append((*res[0], time.monotonic()))
    after = threads.settle(before)
    for i, (st, _, data, dt, _) in enumerate(probes):
        print(f"  {tag}: probe {i}: {st} {data!r} in {dt:.3f} s")
    print(f"  {tag}: futex threads before the probes {before}, while probe 0 waited {during}, after {after}")

    if before is not None:
        if before + 1 not in during:
            problems.append(f"{tag}: no extra futex thread seen while a probe waited ({during}); "
                            "either the probe did not wait or the count cannot see a waiting thread")
        if after != before:
            problems.append(f"{tag}: futex threads {after} after the probes answered, {before} before: "
                            "a failed probe left a thread blocked")
    held = []
    for i, (st, ctype, data, dt, at) in enumerate(probes):
        text = data.decode("utf-8", "replace")
        m = STUCK.fullmatch(text)
        if st != 503 or ctype != "text/plain" or m is None:
            problems.append(f"{tag}: probe {i} answered {st} {ctype!r} {text!r} in {dt:.3f} s, want 503 "
                            "'text/plain' 'stuck lock_wait_ms=N write_held_ms=M'")
            continue
        if dt > lock_ms / 1000 + 0.5:
            problems.append(f"{tag}: probe {i} took {dt:.3f} s, over HEALTH_LOCK_MS + 0.5 s")
        wait, held_for = int(m.group(1)), int(m.group(2))
        if not lock_ms <= wait < lock_ms + 500:
            problems.append(f"{tag}: probe {i} lock_wait_ms={wait}, want {lock_ms} to {lock_ms + 499}")
        if held_for < wait:
            problems.append(f"{tag}: probe {i} write_held_ms={held_for} is less than its lock_wait_ms={wait}")
        held.append((held_for, at))
    for (m0, t0), (m1, t1) in zip(held, held[1:]):
        if abs((m1 - m0) - (t1 - t0) * 1000) > 250:
            problems.append(f"{tag}: write_held_ms went {m0} -> {m1} while {(t1 - t0) * 1000:.0f} ms passed")


def stuck_checks(server, lock_ms, idle_perf, idle_files, problems, skipped, require_hook):
    """The stall sequence on a server whose HEALTH_LOCK_MS is lock_ms."""
    tag = f"stuck (HEALTH_LOCK_MS {lock_ms})"
    threads = Threads(server, problems)
    idle = threads.futex_count()
    box = harness.box(0, 32, 0, 32, 0, 32)
    body = np.full(32 * 32 * 32, 3, dtype=np.uint16).tobytes()
    hooked = sum(HOOK_LINE in line for line in log_lines(server))
    server.exec("touch", MARKER)
    patch = []
    sender = threading.Thread(target=lambda: patch.append(
        get(server, f"/{DS}+token={fixtures.TOKEN}/write/1/{box}", 120, "PATCH", body)))
    sender.start()
    try:
        held = False
        deadline = time.monotonic() + 10
        while not held and sender.is_alive() and time.monotonic() < deadline:
            held = sum(HOOK_LINE in line for line in log_lines(server)) > hooked
            time.sleep(0.1)
        if held:
            held_checks(server, tag, lock_ms, threads, problems)
        elif sender.is_alive():
            problems.append(f"{tag}: the PATCH neither answered nor reached the hook within 10 s")
        else:
            print(f"  {tag}: SKIPPED: this build has no test hook (the PATCH answered {patch[0][0]} in "
                  f"{patch[0][3]:.3f} s with {MARKER} present; build with -DNTRACER_TEST_HOOKS=ON)")
            skipped.append(tag)
            if require_hook:
                problems.append(f"{tag}: skipped, and --require-hook was given")
    finally:
        server.exec("rm", "-f", MARKER)
        sender.join(130)
    if not patch:
        problems.append(f"{tag}: the PATCH did not answer within 130 s of the release")
        return

    st, _, data, dt = patch[0]
    print(f"  {tag}: released; PATCH {st} {data!r} after {dt:.3f} s")
    if st != 200:
        problems.append(f"{tag}: the PATCH answered {st} {data!r}")
    back = get(server, f"/{DS}+token={fixtures.TOKEN}/1/{box}", 10)
    if back[0] != 200 or back[2] != body:
        problems.append(f"{tag}: read-back answered {back[0]} with {len(back[2])} bytes, not what was sent")
    perf_after = get(server, "/performance", 5)[2]
    st, text = check_ok(server, f"{tag}, after release", problems)
    after = threads.settle(idle)
    print(f"  {tag}: after release /health {st} {text!r}, futex threads {after} (idle {idle})")
    if after != idle:
        problems.append(f"{tag}: futex threads {after} after release, {idle} at idle")
    # The no-change checks in idle_checks must be able to fail: the PATCH
    # changed a file and the read added a row to /performance
    if files(server) == idle_files or perf_after == idle_perf:
        problems.append(f"{tag}: the file list or /performance did not change after a PATCH and a read; "
                        "the idle no-change checks cannot fail")


def env_checks(image, platform, data, problems):
    for value, ignored in (("1", False), ("30000", False), ("abc", True), ("0", True), ("30001", True)):
        server = harness.Server(f"health-env{value}", image, platform, data, {"HEALTH_LOCK_MS": value})
        try:
            server.start()
            st, text = check_ok(server, f"HEALTH_LOCK_MS={value}", problems)
            lines = [line for line in log_lines(server) if "HEALTH_LOCK_MS" in line]
        finally:
            server.stop()
            server.remove()
        print(f"  HEALTH_LOCK_MS={value}: /health {st} {text!r}; log {lines}")
        want = [f"HEALTH_LOCK_MS ignored (not a whole number from 1 to 30000): {value}; using {DEFAULT_LOCK_MS}"]
        if lines != (want if ignored else []):
            problems.append(f"HEALTH_LOCK_MS={value}: log lines {lines}, want {want if ignored else []}")


def one_run(n, args, work):
    problems, skipped = [], []
    data = {}
    for role in ("default", "lockms", "readonly"):
        data[role] = os.path.join(work, f"run{n}", role)
        fixtures.segmentation(os.path.join(data[role], DS), (64, 64, 32), (64, 64, 32), harness.RES)
    for role, env in (("default", {}), ("lockms", {"HEALTH_LOCK_MS": str(args.lock_ms)}),
                      ("readonly", {"READ_ONLY": "1"})):
        server = harness.Server(f"health-{role}{n}", args.image, args.platform, data[role], env)
        try:
            server.start()
            print(f" {role} server ({env or 'no settings'}):")
            perf, before = idle_checks(server, problems)
            if role != "readonly":
                lock_ms = args.lock_ms if env else DEFAULT_LOCK_MS
                stuck_checks(server, lock_ms, perf, before, problems, skipped, args.require_hook)
            alive, code = server.running()
            if not alive:
                problems.append(f"{role} server died (exit {code})")
        finally:
            server.stop()
            server.save_logs(os.path.join(work, f"run{n}-{role}.log"))
            server.remove()
    env_checks(args.image, args.platform, data["readonly"], problems)
    return problems, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--platform")
    ap.add_argument("--work", required=True)
    ap.add_argument("--lock-ms", type=int, default=700, help="HEALTH_LOCK_MS of the second stuck server")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--require-hook", action="store_true", help="fail when the build has no test hook")
    args = ap.parse_args()
    if args.lock_ms == DEFAULT_LOCK_MS or not 1 <= args.lock_ms <= 30000:
        ap.error(f"--lock-ms must be from 1 to 30000 and not {DEFAULT_LOCK_MS}, the default it is told apart from")

    work = os.path.abspath(args.work)
    shutil.rmtree(work, ignore_errors=True)
    failed = skipped_runs = 0
    for n in range(args.runs):
        print(f"run {n}: image {args.image}")
        problems, skipped = one_run(n, args, work)
        print(f"run {n}: {len(problems)} problems, {len(skipped)} of 2 held checks SKIPPED")
        for p in problems:
            print(f"  PROBLEM {p}")
        failed += bool(problems)
        skipped_runs += bool(skipped)
    note = f"; held checks SKIPPED in {skipped_runs} (no test hook)" if skipped_runs else ""
    print(f"RESULT: {failed} of {args.runs} runs found a problem{note}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
