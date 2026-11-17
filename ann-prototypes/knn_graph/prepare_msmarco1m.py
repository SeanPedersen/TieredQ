"""Seeded 1M-row float32 subset of the Qwen/MS MARCO SQ8 store (vector = codes * scales + offsets), for #56 at 1M.

    uv run prepare_msmarco1m.py STORE_DIR OUT.npy [--rows 1000000] [--dimensions 1024]
"""
import argparse
from pathlib import Path

import numpy as np

SUBSET_SEED = 56
CHUNK_ROWS = 65536

parser = argparse.ArgumentParser()
parser.add_argument("store", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--rows", type=int, default=1_000_000)
parser.add_argument("--dimensions", type=int, default=1024)
args = parser.parse_args()
codes = np.memmap(args.store / "base.i8", dtype=np.int8, mode="r").reshape(-1, args.dimensions)
scales = np.fromfile(args.store / "scales.f32", np.float32)
offsets = np.fromfile(args.store / "offsets.f32", np.float32)
picked = np.sort(np.random.default_rng(SUBSET_SEED).choice(len(codes), args.rows, replace=False))
out = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float32, shape=(args.rows, args.dimensions))
for start in range(0, args.rows, CHUNK_ROWS):
    part = picked[start : start + CHUNK_ROWS]
    out[start : start + len(part)] = codes[part].astype(np.float32) * scales + offsets
out.flush()
norms = np.linalg.norm(out[:1000], axis=1)
print(f"{args.rows} x {args.dimensions} from {len(codes)} rows; norms of the first 1000: {norms.min():.3f}-{norms.max():.3f}")
