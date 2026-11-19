"""Writes the SuperEMeans repro inputs for the first N Qwen/MS MARCO rows: raw fp32 base and the 1,000 queries
(750 test + 250 calibration) as data_<name>.bin / data_<name>_test.bin, and their exact top-100 as gt_<name>.json.
Usage: prepare.py <rows> <name> <out_dir>"""
import json
import sys
from pathlib import Path

import numpy as np

EXPORT = Path("/mnt/storage/tieredq-issue44-msmarco8m-20261107/export")
SOURCE = "/mnt/storage/tieredq-issue44-msmarco8m-20261107/xet-download/msmarco-qwen-1024-normalized.hdf5"
PAYLOAD_OFFSET, TOTAL_ROWS, DIM, TOP_K, BLOCK_ROWS = 6144, 8840823, 1024, 100, 262144


def read_fvecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    return raw.reshape(-1, DIM + 1)[:, 1:].view(np.float32)


def main(rows, name, out_dir):
    out = Path(out_dir)
    base = np.memmap(SOURCE, dtype=np.float32, mode="r", offset=PAYLOAD_OFFSET, shape=(TOTAL_ROWS, DIM))[:rows]
    np.ascontiguousarray(base).tofile(out / f"data_{name}.bin")
    queries = np.vstack([read_fvecs(EXPORT / "test_queries.fvecs"), read_fvecs(EXPORT / "calibration_queries.fvecs")])
    queries.astype(np.float32).tofile(out / f"data_{name}_test.bin")

    base = np.fromfile(out / f"data_{name}.bin", dtype=np.float32).reshape(rows, DIM)
    best_d = np.full((len(queries), TOP_K), np.inf, dtype=np.float32)
    best_i = np.zeros((len(queries), TOP_K), dtype=np.int64)
    for first in range(0, rows, BLOCK_ROWS):
        block = base[first:first + BLOCK_ROWS]
        distances = (block * block).sum(1)[None, :] - 2.0 * queries @ block.T
        top = np.argpartition(distances, TOP_K, axis=1)[:, :TOP_K]
        merged_d = np.hstack([best_d, np.take_along_axis(distances, top, 1)])
        merged_i = np.hstack([best_i, top + first])
        keep = np.argsort(merged_d, axis=1)[:, :TOP_K]
        best_d, best_i = np.take_along_axis(merged_d, keep, 1), np.take_along_axis(merged_i, keep, 1)
    truth = {str(q): best_i[q].tolist() for q in range(len(queries))}
    (out / f"gt_{name}.json").write_text(json.dumps(truth))
    print(f"{name}: {rows} rows, {len(queries)} queries")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3])
