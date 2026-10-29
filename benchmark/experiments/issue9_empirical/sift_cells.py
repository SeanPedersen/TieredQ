"""Measure SIFT TieredQ cells (1M, L2) at given code bits, seed 0, with measure_memory.run_cell; run from a tree's benchmark dir.

Usage: python sift_cells.py OUTPUT_DIR DATA_DIR CODE_BITS...
"""
import sys
from pathlib import Path
from measure_memory import prepare_native, run_cell
env, _ = prepare_native()
output, data = Path(sys.argv[1]), sys.argv[2]
output.mkdir(parents=True, exist_ok=True)
for code_bits in map(int, sys.argv[3:]):
    row = run_cell(dict(dataset="sift", codec="tieredq", bits=1, seed=0, data=data, quick=True, code_bits=code_bits), output, env)
    print("RESULT", code_bits, row["disk_bits_per_vector"], row["recall10"], flush=True)
