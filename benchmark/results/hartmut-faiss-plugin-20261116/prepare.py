"""Writes the acceptance benchmark's data directories (meta.txt, queries.f32, truth.i32, base): VIBE arxiv as float32
rows from its hdf5 (neighbors: top 100 by inner product), Qwen/MS MARCO 8.84M as links to its SQ8 store with the
held-out queries and their top 10. Usage: prepare.py OUT"""
import os
import sys
from pathlib import Path

import h5py
import numpy as np

QWEN = Path("/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107")
ARXIV = Path("/home/sean/experiments/TieredQ-vibe-arxiv-20261106/data/arxiv-nomic-768-normalized.hdf5")


def vecs(path, dtype):
    raw = np.fromfile(path, dtype=np.int32)
    return np.ascontiguousarray(raw.reshape(-1, raw[0] + 1)[:, 1:]).view(dtype)


out = Path(sys.argv[1])
arxiv = out / "arxiv"
arxiv.mkdir(parents=True, exist_ok=True)
if not (arxiv / "meta.txt").exists():
    with h5py.File(ARXIV, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        train.tofile(arxiv / "base.f32")
        np.asarray(f["test"], dtype=np.float32).tofile(arxiv / "queries.f32")
        np.asarray(f["neighbors"], dtype=np.int32).tofile(arxiv / "truth.i32")
    (arxiv / "meta.txt").write_text(f"{train.shape[0]} {train.shape[1]} 100 f32\n")
qwen = out / "qwen"
qwen.mkdir(parents=True, exist_ok=True)
if not (qwen / "meta.txt").exists():
    for name in ("base.i8", "scales.f32", "offsets.f32"):
        if not (qwen / name).exists():
            os.symlink(QWEN / name, qwen / name)
    queries = vecs(QWEN / "data/test_queries.fvecs", np.float32)
    queries.tofile(qwen / "queries.f32")
    vecs(QWEN / "data/test_truth.ivecs", np.int32)[:, :10].astype(np.int32).tofile(qwen / "truth.i32")
    rows = (QWEN / "base.i8").stat().st_size // queries.shape[1]
    (qwen / "meta.txt").write_text(f"{rows} {queries.shape[1]} 10 sq8\n")
