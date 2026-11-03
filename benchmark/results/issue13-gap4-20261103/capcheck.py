"""Seed-0 R@10 (8t, batch 32) with TQ_MAX_BITS from the env, at given code budgets: is the 6-bit cap a no-op there?"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.environ["ER_DIR"])
from equal_recall import recall_at
from data import prepare
import codec_native
codec_native.set_threads(8)
dataset, data = sys.argv[1], Path(sys.argv[2])
base, queries, truth = prepare(data, dataset)
for code in map(int, sys.argv[3:]):
    print(dataset, os.environ["TQ_MAX_BITS"], code, f"{recall_at(base, queries, truth, False, code, 0):.4f}", flush=True)
