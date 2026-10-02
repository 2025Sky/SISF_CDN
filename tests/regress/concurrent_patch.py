"""Runs harness scenario s19 (PATCHes sent at once into one mchunk) on one
image, each run on a fresh server and dataset, and prints every problem.

    python concurrent_patch.py --image IMAGE --work DIR [--platform linux/amd64]
                               [--rounds 40] [--runs 1]

In the harness s19 is compared with the baseline like any other case. Here it
is judged on its own, so a build can be checked without a baseline: exit
status 1 if any run had a PATCH that did not answer 200, a chunk that read
back with other contents or did not answer 200, a .meta entry outside the
.data, sharing bytes with another or not decoding to the chunk sent last, or
a server that died.
"""

import argparse
import os
import shutil
import sys

import harness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--platform")
    ap.add_argument("--work", required=True)
    ap.add_argument("--rounds", type=int, default=harness.CPATCH_ROUNDS)
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    work = os.path.abspath(args.work)
    shutil.rmtree(work, ignore_errors=True)
    failed = 0
    for n in range(args.runs):
        data = os.path.join(work, f"run{n}")
        os.makedirs(data)
        server = harness.Server(f"cpatch{n}", args.image, args.platform, data)
        results, problems = {}, []
        try:
            server.start()
            harness.sc_concurrent_patch(server, results, "s19", rounds=args.rounds, problems=problems)
            alive, code = server.running()
        finally:
            server.stop()
            server.save_logs(os.path.join(work, f"run{n}.log"))
            server.remove()
        if not alive:
            problems.append(f"server died (exit {code})")
        print(f"run {n}: image {args.image}, {args.rounds} rounds: {len(problems)} problems")
        for key, r in results.items():
            print(f"  {key}: {r['status']}: {r['text'] if r['sha256'] is None else r['sha256']}")
        for p in problems:
            print(f"  PROBLEM {p}")
        failed += bool(problems)
    print(f"RESULT: {failed} of {args.runs} runs found a problem")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
