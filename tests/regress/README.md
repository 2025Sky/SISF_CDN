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
- After the write and danger stages, `scenarios` runs the cases that kill or
  hang production, each on a dataset it builds while the server runs, with a
  liveness check after each: stale archive geometry after an in-place
  re-conversion (the 2026-08-30 abort), a corrupt zstd frame, an unknown
  compression type in an mchunk header, a writable layer with a missing
  `.data` file (production leaks its write lock; that case uses a 20 s client
  timeout and restarts a server that timed out), and a PATCH over a chunk
  whose data is cut short (production overwrites the chunk's other voxels
  with zeros and answers 200).
- `expected_diffs.json` lists the differences that are intended, each with a
  reason. Anything else fails the run, and so does any candidate CRASH or
  TIMEOUT, even where the baseline crashes the same way, and so does a listed
  difference that did not occur (the candidate behaves like the baseline
  there again).
- Each server's full log is saved to `<work>/<role>.log`.

Compare like with like: run both images on amd64. An integer division by zero
that kills the process on x86-64 returns 0 on arm64, so an arm64 candidate can
pass a case that crashes in production.
