"""Generate small SISF datasets for the black-box regression harness.

The byte layout follows pySISF 0.4.1 (``create_metadata`` / ``create_shard`` /
``create_sisf``) and the SAVAII portal's tiled writer, which is how production
data is written. Only numpy and zstd are needed, so CI does not have to install
pySISF's heavier dependencies.

Usage: python fixtures.py OUT_DIR
"""

import json
import os
import struct
import sys

import numpy as np
import zstd

HEADER_LAYOUT = "<" + "H" * 6 + "Q" * 6  # metadata.bin, 60 bytes
SHARD_HEADER_LAYOUT = "<" + "H" * 7 + "Q" * 9  # .meta header, 86 bytes
SHARD_LINE_LAYOUT = "<QL"  # one table entry: offset, compressed size
VERSION = 1
DTYPE_U16 = 1
COMP_ZSTD = 1
TOKEN = "regress-token"


def iterate_bounded(n, step):
    i = 0
    while i < n:
        yield i, min(i + step, n)
        i += step


def pattern(shape, seed):
    """Deterministic uint16 volume: a smooth ramp plus seeded noise, so chunks
    compress to different sizes and every voxel value depends on its position."""
    rng = np.random.default_rng(seed)
    grids = np.indices(shape, dtype=np.uint32)
    ramp = grids[0] * 7 + grids[1] * 131 + grids[-1] * 1031 + seed * 17
    noise = rng.integers(0, 64, size=shape, dtype=np.uint32)
    return ((ramp + noise) % 65536).astype(np.uint16)


def write_metadata(out, channels, mchunk, res, size):
    os.makedirs(f"{out}/meta", exist_ok=True)
    os.makedirs(f"{out}/data", exist_ok=True)
    with open(f"{out}/metadata.bin", "wb") as f:
        f.write(struct.pack(HEADER_LAYOUT, VERSION, DTYPE_U16, channels, *mchunk, *res, *size))


def create_shard(fdata, fmeta, data, chunk, crop=None):
    """Same bytes as pySISF.sisf.create_shard(..., compression=1)."""
    table, blob = [], bytearray()
    for i0, i1 in iterate_bounded(data.shape[0], chunk[0]):
        for j0, j1 in iterate_bounded(data.shape[1], chunk[1]):
            for k0, k1 in iterate_bounded(data.shape[2], chunk[2]):
                comp = zstd.ZSTD_compress(data[i0:i1, j0:j1, k0:k1].tobytes(order="C"), 9, 1)
                table.append((len(blob), len(comp)))
                blob += comp
    if crop is None:
        crop = (0, data.shape[0], 0, data.shape[1], 0, data.shape[2])
    with open(fdata, "wb") as f:
        f.write(blob)
    with open(fmeta, "wb") as f:
        f.write(struct.pack(SHARD_HEADER_LAYOUT, VERSION, DTYPE_U16, 1, COMP_ZSTD, *chunk, *data.shape, *crop))
        for off, n in table:
            f.write(struct.pack(SHARD_LINE_LAYOUT, off, n))


def write_skeleton(fdata, fmeta, shape, chunk):
    """An mchunk with no data yet, as the portal writes empty segmentations:
    every table entry zero and an empty (but existing) .data file."""
    n = 1
    for a in range(3):
        n *= len(list(iterate_bounded(shape[a], chunk[a])))
    with open(fmeta, "wb") as f:
        f.write(struct.pack(SHARD_HEADER_LAYOUT, VERSION, DTYPE_U16, 1, COMP_ZSTD, *chunk, *shape,
                            0, shape[0], 0, shape[1], 0, shape[2]))
        f.write(b"\x00" * (n * struct.calcsize(SHARD_LINE_LAYOUT)))
    open(fdata, "wb").close()


def downsample2(a):
    """2x2x2 block mean (floor), matching pySISF.sndif_utils.downsample."""
    s = tuple(d // 2 for d in a.shape)
    v = a[: s[0] * 2, : s[1] * 2, : s[2] * 2].astype(np.uint32)
    v = v.reshape(s[0], 2, s[1], 2, s[2], 2).sum(axis=(1, 3, 5)) // 8
    return v.astype(np.uint16)


def untiled(out, vol_cxyz, mchunk, res, levels):
    """pySISF.create_sisf layout: mchunks cut at 1X, each downsampled on its own."""
    channels, size = vol_cxyz.shape[0], vol_cxyz.shape[1:]
    write_metadata(out, channels, mchunk, res, size)
    for c in range(channels):
        for i, (i0, i1) in enumerate(iterate_bounded(size[0], mchunk[0])):
            for j, (j0, j1) in enumerate(iterate_bounded(size[1], mchunk[1])):
                for k, (k0, k1) in enumerate(iterate_bounded(size[2], mchunk[2])):
                    block = np.ascontiguousarray(vol_cxyz[c, i0:i1, j0:j1, k0:k1])
                    for lvl in range(levels):
                        if lvl:
                            block = downsample2(block)
                        name = f"chunk_{i}_{j}_{k}.{c}.{2 ** lvl}X"
                        create_shard(f"{out}/data/{name}.data", f"{out}/meta/{name}.meta", block, (32, 32, 32))


def tiled(out, channels, grid, tile, overlap_px, res, seed):
    """Portal tiled layout: each tile is one mchunk holding the whole tile, with
    a crop window that trims half the overlap on each side; metadata.bin records
    the tile step as the mchunk size."""
    step = tuple(t - o for t, o in zip(tile, overlap_px))
    write_metadata(out, channels, step, res, tuple(g * s for g, s in zip(grid, step)))
    margin = tuple(o // 2 for o in overlap_px)
    crop6 = (margin[0], overlap_px[0] - margin[0], margin[1], overlap_px[1] - margin[1],
             margin[2], overlap_px[2] - margin[2])
    min_xy = 32 if tile[2] == 1 else 256
    for c in range(channels):
        for tx in range(grid[0]):
            for ty in range(grid[1]):
                for tz in range(grid[2]):
                    cur = pattern(tile, seed + 101 * c + 11 * tx + 7 * ty + 3 * tz)
                    base = cur
                    for lvl in range(10):
                        if lvl:
                            shape = tuple(max(1, d // 2) for d in cur.shape)
                            if any(d < min_xy for d in shape[:2]):
                                break
                            cur = downsample2(cur) if cur.shape[2] > 1 else downsample2_xy(cur)
                        f = tuple(max(1, b // d) for b, d in zip(base.shape, cur.shape))
                        l, r, t, b, fr, bk = (crop6[0] // f[0], crop6[1] // f[0], crop6[2] // f[1],
                                              crop6[3] // f[1], crop6[4] // f[2], crop6[5] // f[2])
                        crop = (l, cur.shape[0] - r, t, cur.shape[1] - b, fr, cur.shape[2] - bk)
                        chunk = tuple(min(d, 32) for d in cur.shape)
                        name = f"chunk_{tx}_{ty}_{tz}.{c}.{2 ** lvl}X"
                        create_shard(f"{out}/data/{name}.data", f"{out}/meta/{name}.meta", cur, chunk, crop)


def downsample2_xy(a):
    """Block mean over x and y only, for single-plane tiles (z stays 1)."""
    s = (a.shape[0] // 2, a.shape[1] // 2, a.shape[2])
    v = a[: s[0] * 2, : s[1] * 2, :].astype(np.uint32)
    v = v.reshape(s[0], 2, s[1], 2, s[2]).sum(axis=(1, 3)) // 4
    return v.astype(np.uint16)


def segmentation(out, size, mchunk, res, prefill_seed=None):
    """Single-channel writable layer. With no seed every mchunk is a skeleton
    (the portal's empty segmentation); with a seed it starts with labels."""
    write_metadata(out, 1, mchunk, res, size)
    for i, (i0, i1) in enumerate(iterate_bounded(size[0], mchunk[0])):
        for j, (j0, j1) in enumerate(iterate_bounded(size[1], mchunk[1])):
            for k, (k0, k1) in enumerate(iterate_bounded(size[2], mchunk[2])):
                name = f"chunk_{i}_{j}_{k}.0.1X"
                shape = (i1 - i0, j1 - j0, k1 - k0)
                if prefill_seed is None:
                    write_skeleton(f"{out}/data/{name}.data", f"{out}/meta/{name}.meta", shape, (32, 32, 32))
                else:
                    labels = (pattern(shape, prefill_seed + i * 9 + j * 5 + k) % 7).astype(np.uint16)
                    create_shard(f"{out}/data/{name}.data", f"{out}/meta/{name}.meta", labels, (32, 32, 32))
    with open(f"{out}/.sisf_access", "w") as f:
        f.write(TOKEN + "\n")


def plane(out, size_xy, res, seed):
    """Untiled single-plane image in the portal's layout: one mchunk, z stays 1
    down the pyramid, levels stop before x or y drops under 32. At 2X the
    scaled mchunk depth (1 // 2) is 0, the case Bin's "support single slice"
    patch clamps; unclamped it is an integer division by zero, which traps on
    x86-64 but silently yields 0 on arm64."""
    size = (size_xy[0], size_xy[1], 1)
    write_metadata(out, 1, size, res, size)
    cur = pattern(size, seed)
    for lvl in range(10):
        if lvl:
            if min(cur.shape[0] // 2, cur.shape[1] // 2) < 32:
                break
            cur = downsample2_xy(cur)
        name = f"chunk_0_0_0.0.{2 ** lvl}X"
        create_shard(f"{out}/data/{name}.data", f"{out}/meta/{name}.meta", cur, tuple(min(d, 32) for d in cur.shape))


def pointclouds(parent, size):
    d = f"{parent}/pointclouds"
    os.makedirs(d, exist_ok=True)
    rows = "\n".join(f"{(i * 13) % size[0]},{(i * 7) % size[1]},{(i * 3) % size[2]},{i * 0.25},{i % 4}"
                     for i in range(40))
    for name, extra in (("pc_new", {"dimensions": {"x": [6.5e-7, "m"], "y": [6.5e-7, "m"], "z": [1.5e-6, "m"]}}),
                        ("pc_old", {"resolution": [650, 650, 1500]})):
        with open(f"{d}/{name}.json", "w") as f:
            json.dump({"size": list(size), **extra}, f)
        with open(f"{d}/{name}.csv", "w") as f:
            f.write("x,y,z,score,label\n" + rows + "\n")
    # Metadata with neither key: the CDN must refuse it the same way.
    with open(f"{d}/pc_nodims.json", "w") as f:
        json.dump({"size": list(size)}, f)
    with open(f"{d}/pc_nodims.csv", "w") as f:
        f.write("x,y,z\n1,2,3\n")


def main(out):
    os.makedirs(out, exist_ok=True)
    untiled(f"{out}/vol1c", pattern((1, 100, 90, 37), 1), (64, 64, 32), (650, 650, 1500), 2)
    untiled(f"{out}/vol3c", pattern((3, 70, 60, 20), 2), (70, 60, 20), (650, 650, 1500), 2)
    tiled(f"{out}/tiled2d", 2, (3, 2, 1), (96, 96, 1), (24, 24, 0), (650, 650, 1500), 3)
    tiled(f"{out}/tiled3d", 1, (2, 2, 2), (64, 64, 40), (8, 8, 4), (650, 650, 1500), 4)
    segmentation(f"{out}/seg_empty", (160, 150, 70), (96, 96, 48), (650, 650, 1500))
    segmentation(f"{out}/seg_prefilled", (100, 90, 37), (64, 64, 32), (650, 650, 1500), prefill_seed=5)
    pointclouds(f"{out}/vol1c", (100, 90, 37))
    plane(f"{out}/plane2d", (200, 150), (650, 650, 1500), 6)
    # A writable layer with one mchunk missing: reads there answer zeros, and a
    # write that reaches it exercises the write path's missing-mchunk branch.
    segmentation(f"{out}/seg_holes", (160, 150, 70), (96, 96, 48), (650, 650, 1500))
    for ext in ("data", "meta"):
        os.remove(f"{out}/seg_holes/{ext}/chunk_1_0_0.0.1X.{ext}")


if __name__ == "__main__":
    main(sys.argv[1])
