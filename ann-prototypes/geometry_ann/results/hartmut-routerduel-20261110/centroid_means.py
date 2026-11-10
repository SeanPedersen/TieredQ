"""Leaf centroids as the mean of each list's SQ8-decoded rows (what the index routes with), from a label file."""
import argparse
from pathlib import Path
import numpy as np

CHUNK_ROWS = 1 << 18


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("codes", "scales", "offsets", "labels", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    labels = np.fromfile(args.labels, dtype="<u4")
    codes = np.memmap(args.codes, dtype=np.int8, mode="r", shape=(labels.size, args.dim))
    scales, offsets = np.fromfile(args.scales, dtype="<f4"), np.fromfile(args.offsets, dtype="<f4")
    sums = np.zeros((args.k, args.dim), dtype=np.float64)
    counts = np.bincount(labels, minlength=args.k).astype(np.float64)
    for start in range(0, labels.size, CHUNK_ROWS):
        chunk_labels = labels[start:start + CHUNK_ROWS]
        order = np.argsort(chunk_labels, kind="stable")
        rows = codes[start:start + CHUNK_ROWS][order].astype(np.float32) * scales + offsets
        sorted_labels = chunk_labels[order]
        firsts = np.flatnonzero(np.r_[True, sorted_labels[1:] != sorted_labels[:-1]])
        sums[sorted_labels[firsts]] += np.add.reduceat(rows, firsts, axis=0)
    if (counts == 0).any():
        raise ValueError("empty list")
    (sums / counts[:, None]).astype("<f4").tofile(args.output)


if __name__ == "__main__":
    main()
