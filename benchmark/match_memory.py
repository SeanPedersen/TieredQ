"""Equal-memory TieredQ vs a reference codec: calibrate TieredQ's code-bit budget to each measured reference point.

The reference (RaBitQ by default, or Qdrant TQ+ with --reference qdrant) is measured at every requested bits/dim
and seed. Each (reference bits, storage metric) pair is one target: the minimum of that metric over its seeds. For each target, the search finds the largest TieredQ `code_bits`
whose seed-0 measurement is at or below the target. It brackets the target, then bisects with secant steps down
to RESOLUTION code bits. Selection reads storage only, never recall. Every measured cell is shared across
targets (and across references sharing an output directory) and logged in calibration.csv. Each chosen budget is then measured at the remaining seeds. If any seed
exceeds the target, the search steps down to the next smaller calibrated budget that fits. Cells are measured
by measure_memory.run_cell, so methodology and columns match memory.csv, and cells.csv makes the run resumable.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time

from measure_memory import FIELDS, HERE, prepare_native, run_cell

METRICS = {"disk": "disk_bits_per_vector", "ram": "ram_loaded_bits_per_vector", "peak": "ram_peak_bits_per_vector"}
RESOLUTION = 8
MAX_PROBES = 24
CALIBRATION_SEED = 0
CALIBRATION_FIELDS = ["dataset", "reference_codec", "reference_bits", "metric", "target_bits_per_vector", "probe", "code_bits",
                      "realised_code_bits", "measured_bits_per_vector", "fits", "seed"]


class Cells:
    """Measured rows keyed by configuration; persisted to cells.csv so an interrupted run resumes."""

    def __init__(self, output, env, data, quick):
        self.path, self.env, self.data, self.output, self.quick = output / "cells.csv", env, data, output, quick
        self.rows = {}
        if self.path.exists():
            with self.path.open() as handle:
                for row in csv.DictReader(handle):
                    self.rows[self.key(row["dataset"], row["codec"], row["target_bits"],
                                       row["requested_code_bits"], row["seed"])] = row

    @staticmethod
    def key(dataset, codec, bits, code_bits, seed):
        return dataset, codec, int(bits), int(code_bits or 0) if codec == "tieredq" else 0, int(seed)

    def measure(self, dataset, codec, bits, seed, code_bits=0):
        key = self.key(dataset, codec, bits, code_bits, seed)
        if key in self.rows:
            return self.rows[key]
        spec = dict(dataset=dataset, codec=codec, bits=bits, seed=seed, data=str(self.data), quick=self.quick)
        if code_bits:
            spec["code_bits"] = code_bits
        row = {name: str(value) for name, value in run_cell(spec, self.output, self.env).items()}
        # The requested budget is the key; the realised one stays in the row for auditing.
        row["requested_code_bits"] = str(code_bits)
        self.rows[key] = row
        new_file = not self.path.exists()
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS + ["requested_code_bits"])
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        return row


def next_probe(fitting, failing, target, dimensions):
    """Next budget inside the open bracket, or None once it is narrower than RESOLUTION.

    fitting/failing map code_bits to the measured metric. Storage grows about one bit per code bit, so a
    one-sided bracket steps by the measured gap and a two-sided one interpolates between its ends.
    """
    upper_limit = 8 * dimensions
    lo = max(fitting, default=None)
    hi = min((code for code in failing if lo is None or code > lo), default=None)
    if lo is not None and hi is not None and hi - lo <= RESOLUTION:
        return None
    if lo is None and hi is None:
        return max(1, min(upper_limit, round(target)))
    if lo is None:
        if hi == 1:
            return None
        return max(1, min(hi - RESOLUTION, round(hi - (failing[hi] - target) * 1.05)))
    if hi is None:
        if lo == upper_limit:
            return None
        return min(upper_limit, max(lo + RESOLUTION, round(lo + (target - fitting[lo]))))
    slope = (failing[hi] - fitting[lo]) / (hi - lo)
    guess = lo + (target - fitting[lo]) / slope if slope > 0 else (lo + hi) / 2
    # Clamp away from the ends so a poor secant still halves the bracket within a few steps.
    margin = max(1, (hi - lo) // 4)
    return round(min(hi - margin, max(lo + margin, guess)))


def calibrate(cells, dataset, reference, metric, target, dimensions, log):
    """Largest seed-0 TieredQ budget with metric ≤ target, or None if even the smallest budget overshoots."""
    column = METRICS[metric]
    fitting, failing = {}, {}
    for probe in range(MAX_PROBES):
        # Budgets already measured for other targets enter the bracket without a new run.
        for (data_set, codec, _, code, seed), row in cells.rows.items():
            if data_set == dataset and codec == "tieredq" and seed == CALIBRATION_SEED and code:
                (fitting if float(row[column]) <= target else failing)[code] = float(row[column])
        code_bits = next_probe(fitting, failing, target, dimensions)
        if code_bits is None:
            break
        row = cells.measure(dataset, "tieredq", -(-code_bits // dimensions), CALIBRATION_SEED, code_bits)
        log(reference | dict(dataset=dataset, metric=metric, target_bits_per_vector=target,
                 probe=probe, code_bits=code_bits, realised_code_bits=row["code_bits"],
                 measured_bits_per_vector=row[column], fits=float(row[column]) <= target, seed=CALIBRATION_SEED))
    else:
        raise RuntimeError(f"{dataset} {reference} {metric}: no bracket within {MAX_PROBES} probes")
    return max(fitting, default=None), sorted(fitting, reverse=True)


def confirm(cells, dataset, metric, target, candidates, dimensions, seeds, log, reference):
    """First candidate (largest first) whose every seed fits the target; returns its budget and rows."""
    column = METRICS[metric]
    for code_bits in candidates:
        rows = [cells.measure(dataset, "tieredq", -(-code_bits // dimensions), seed, code_bits) for seed in seeds]
        for row in rows:
            log(reference | dict(dataset=dataset, metric=metric, target_bits_per_vector=target,
                     probe="confirm", code_bits=code_bits, realised_code_bits=row["code_bits"],
                     measured_bits_per_vector=row[column], fits=float(row[column]) <= target, seed=row["seed"]))
        if all(float(row[column]) <= target for row in rows):
            return code_bits, rows
    return None, []


def summary(rows, column):
    values = [float(row[column]) for row in rows]
    return dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values) > 1 else 0.0,
                max=max(values), min=min(values))


def match(cells, dataset, codec, bits, metric, seeds, log):
    measured = [cells.measure(dataset, codec, bits, seed) for seed in seeds]
    dimensions = int(measured[0]["d"])
    target = min(float(row[METRICS[metric]]) for row in measured)
    reference = dict(reference_codec=codec, reference_bits=bits)
    best, candidates = calibrate(cells, dataset, reference, metric, target, dimensions, log)
    chosen, tieredq = confirm(cells, dataset, metric, target, candidates, dimensions, seeds, log, reference)
    result = reference | dict(dataset=dataset, metric=metric, target_bits_per_vector=target,
                              calibrated_code_bits=best, code_bits=chosen,
                              reference={name: summary(measured, column) for name, column in METRICS.items()}
                              | {"recall10": summary(measured, "recall10")})
    if chosen is None:
        return result | dict(tieredq=None, win=False)
    stats = {name: summary(tieredq, column) for name, column in METRICS.items()} | {
        "recall10": summary(tieredq, "recall10"), "realised_code_bits": int(tieredq[0]["code_bits"])}
    win = stats[metric]["max"] <= target and stats["recall10"]["mean"] > result["reference"]["recall10"]["mean"]
    return result | dict(tieredq=stats, win=win)


REFERENCE_NAMES = {"rabitq": "RaBitQ", "qdrant": "Qdrant TQ+"}


def markdown(results):
    name = REFERENCE_NAMES[results[0]["reference_codec"]]
    lines = [f"Bits/vector; recall@10 is mean ± std over seeds. Target = min of {name}'s metric over seeds; "
             "TieredQ memory is the max over seeds.", "",
             f"| Dataset | Matched on | {name} bits/dim | Target | TieredQ code bits | TieredQ disk | TieredQ RAM "
             f"| TieredQ peak | TieredQ R@10 | {name} disk | {name} RAM | {name} peak | {name} R@10 | Δ R@10 | Win |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    recall = lambda stats: f"{stats['mean']:.4f} ± {stats['std']:.4f}"
    for result in results:
        reference, tieredq = result["reference"], result["tieredq"]
        head = (f"| {result['dataset']} | {result['metric']} | {result['reference_bits']} | "
                f"{result['target_bits_per_vector']:,.1f} | ")
        reference_cells = (f"{reference['disk']['max']:,.1f} | {reference['ram']['max']:,.1f} | "
                           f"{reference['peak']['max']:,.1f} | {recall(reference['recall10'])}")
        if tieredq is None:
            lines.append(head + "none fits | – | – | – | – | " + reference_cells + " | – | no |")
            continue
        delta = tieredq["recall10"]["mean"] - reference["recall10"]["mean"]
        lines.append(head + f"{result['code_bits']:,} | {tieredq['disk']['max']:,.1f} | {tieredq['ram']['max']:,.1f} | "
                     f"{tieredq['peak']['max']:,.1f} | {recall(tieredq['recall10'])} | {reference_cells} | "
                     f"{delta:+.4f} | {'yes' if result['win'] else 'no'} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["openai", "gemini"])
    parser.add_argument("--reference", choices=REFERENCE_NAMES, default="rabitq")
    parser.add_argument("--reference-bits", "--rabitq-bits", nargs="+", type=int, help="default: 1-4 (RaBitQ), 1/2/4 (TQ+)")
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, default=HERE / "output-match")
    parser.add_argument("--quick", action="store_true", help="memory-only query workload per cell, no timing")
    args = parser.parse_args()
    args.reference_bits = args.reference_bits or ([1, 2, 4] if args.reference == "qdrant" else [1, 2, 3, 4])
    if args.seeds[0] != CALIBRATION_SEED:
        raise SystemExit(f"--seeds must start with the calibration seed {CALIBRATION_SEED}")
    args.output.mkdir(parents=True, exist_ok=True)
    env, meta = prepare_native()
    cells = Cells(args.output, env, args.data.resolve(), args.quick)
    calibration = (args.output / "calibration.csv").open("a", newline="")
    writer = csv.DictWriter(calibration, fieldnames=CALIBRATION_FIELDS)
    if calibration.tell() == 0:
        writer.writeheader()

    def log(entry):
        writer.writerow(entry)
        calibration.flush()

    start = time.perf_counter()
    results = [match(cells, dataset, args.reference, bits, metric, args.seeds, log)
               for dataset in args.datasets for bits in args.reference_bits for metric in args.metrics]
    calibration.close()
    meta["wall_s"] = time.perf_counter() - start
    # One output directory may hold matches against several references; TieredQ probes in cells.csv are shared.
    (args.output / f"meta-{args.reference}.json").write_text(json.dumps(meta, indent=2))
    (args.output / f"matches-{args.reference}.json").write_text(json.dumps(results, indent=2))
    (args.output / f"matches-{args.reference}.md").write_text(markdown(results))


if __name__ == "__main__":
    main()
