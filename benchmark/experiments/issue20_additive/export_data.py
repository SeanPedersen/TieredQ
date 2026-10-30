"""Export 100k-base / 1k-query raw float32 sets with inner-product top-10 truth on the raw base.

openai1m exports its full 999k base. Usage: python export_data.py DATA_DIR OUTPUT_DIR [DATASET ...]
"""
import sys
from pathlib import Path

import numpy as np

BASE_ROWS, TOP_K, CHUNK = 100_000, 10, 100
SOURCES = {"glove": "glove-100000-1000-v2", "sift": "sift-1000000-1000-v2", "openai": "openai-100000-1000-v2", "gemini": "gemini-prepared",
           "openai1m": "openai1m-999000-1000-v2"}
FULL_BASE = {"openai1m"}


def inner_product_truth(base, queries):
    truth = np.empty((len(queries), TOP_K), dtype=np.uint32)
    for start in range(0, len(queries), CHUNK):
        scores = queries[start:start + CHUNK] @ base.T
        top = np.argpartition(-scores, TOP_K, axis=1)[:, :TOP_K]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1, kind="stable")
        truth[start:start + CHUNK] = np.take_along_axis(top, order, axis=1)
    return truth


def main(data, output, names):
    for name in names:
        arrays = np.load(data / f"{SOURCES[name]}.npz")
        base = np.ascontiguousarray(arrays["base"][:None if name in FULL_BASE else BASE_ROWS], dtype=np.float32)
        queries = np.ascontiguousarray(arrays["queries"], dtype=np.float32)
        target = output / name
        target.mkdir(parents=True, exist_ok=True)
        base.tofile(target / "base.f32")
        queries.tofile(target / "queries.f32")
        inner_product_truth(base, queries).tofile(target / "truth.u32")
        print(name, base.shape, queries.shape, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:] or [name for name in SOURCES if name not in FULL_BASE])
