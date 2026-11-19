"""Recall@k of each log's [assign labels] curve at matched vectors scanned (log-linear interpolation).
Usage: matched.py <log>..."""
import re
import sys

import numpy as np

TARGETS = [5_000, 12_000, 24_000, 60_000]
POINT = re.compile(r"\(\s*[\d.]+% centroids,\s+(\d+) avg vectors\):\s+([\d.]+)")


def curves(path):
    text = open(path).read().split("[assign labels]")[1]
    out = {}
    for k, block in re.findall(r"--- Recall@(\d+) ---\n(.*?)(?=\n\n|\n---|\Z)", text, re.S):
        points = np.array([(float(v), float(r)) for v, r in POINT.findall(block)])
        out[int(k)] = [np.interp(np.log(t), np.log(points[:, 0]), points[:, 1]) for t in TARGETS]
    return out


print("vectors scanned:", TARGETS)
for path in sys.argv[1:]:
    name = path.split("/")[-1].split("-qwen")[0]
    for k, values in curves(path).items():
        print(f"{name:12s} R@{k:<3d}", " ".join(f"{v:.4f}" for v in values))
