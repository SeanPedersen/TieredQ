"""Time TieredQ at every equal-memory matched code budget (seed 0, threads 1/8 x batch 1/32), one fresh process each.

Budgets come from match_memory results (matches-*.json in the MATCH_GLOB directories), plus optionally the
OpenAI/Gemini RaBitQ matches from issue4-equal-memory-20261028/after (PREVIOUS_RABITQ_MATCHES; older key names).
Recall over seeds 0-2 is already in the match results; this adds speed at the same budgets. Budgets with an existing
cell file are reused, so a re-match only times its new budgets.
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / os.environ.get("FINAL_SOURCE", "source-align")
OUTPUT = ROOT / os.environ.get("MATCHED_OUTPUT", "output-matched-timing")
MATCH_GLOB = os.environ.get("MATCH_GLOB", "match-qdrant-*")
DATA = Path(os.environ["FINAL_DATA"])
PREVIOUS_RABITQ = os.environ.get("PREVIOUS_RABITQ_MATCHES")
REFERENCE_BITS = (1, 2, 4)


def matched_budgets():
    budgets = set()
    sources = list(ROOT.glob(f"{MATCH_GLOB}/matches-*.json")) + ([Path(PREVIOUS_RABITQ)] if PREVIOUS_RABITQ else [])
    for path in sources:
        for result in json.loads(path.read_text()):
            # References are compared at 1/2/4 bits/dim; older RaBitQ matches also hold 3-bit targets.
            if result.get("code_bits") and result.get("reference_bits", result.get("rabitq_bits")) in REFERENCE_BITS:
                budgets.add((result["dataset"], int(result["code_bits"])))
    return sorted(budgets)


def main():
    sys.path.insert(0, str(SOURCE / "benchmark"))
    from native_build import build_native
    from run import FIELDS
    OUTPUT.mkdir(exist_ok=True)
    native = build_native()
    env = os.environ | {"PYTHONPATH": str(native.directory), "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "8", "BENCH_SEED": "0"}
    budgets = matched_budgets()
    dimensions = {}
    with (OUTPUT / "matched-timing.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS + ["code_bits"])
        writer.writeheader()
        for number, (dataset, code_bits) in enumerate(budgets, 1):
            cell = OUTPUT / f"cell-{dataset}-c{code_bits}.json"
            # bits only labels the row; the budget is code_bits.
            d = dimensions.setdefault(dataset, {"sift": 128, "glove": 200, "openai": 1536, "gemini": 768}[dataset])
            bits = -(-code_bits // d)
            command = [sys.executable, SOURCE / "benchmark/run.py", "--codec", "tieredq", "--datasets", dataset,
                       "--bits", bits, "--code-bits", code_bits, "--seeds", 0, "--data", DATA, "--output", OUTPUT, "--cell", cell]
            if not cell.exists():
                print("RUN", *map(str, command), flush=True)
                subprocess.run(list(map(str, command)), env=env, check=True, cwd=SOURCE / "benchmark")
            writer.writerows(row | {"code_bits": code_bits} for row in json.loads(cell.read_text()))
            out.flush()
            print(f"PROGRESS {number}/{len(budgets)}", flush=True)
    print("MATCHED TIMING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
