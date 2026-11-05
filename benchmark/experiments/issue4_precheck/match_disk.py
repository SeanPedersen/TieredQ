"""Equal-disk TieredQ budgets for the #4 precheck: per reference cell and TieredQ variant, the largest seed-0 code_bits
whose run.py effective bytes/vector (the same `storage()` measure the results rows report) is at or below the
reference's.

Usage: match_disk.py DATASET DATA OUTPUT REFERENCE_CELL.json...
Reference cells are run.py worker outputs; their effective_bytes_per_vector is the target. Probes are built in process
(no search) with match_memory's bracket search at its RESOLUTION, and every probe is logged to
OUTPUT/calibration-DATASET.csv. OUTPUT/matches-DATASET.json lists {reference, reference_bits, variant, code_bits, ...}.
"""
import argparse
import csv
import json
from pathlib import Path
import tempfile

from match_memory import MAX_PROBES, next_probe
from run import TIEREDQ_VARIANTS, tieredq_index

BITS_PER_BYTE = 8
MAX_BITS_PER_DIM = 8
CALIBRATION_SEED = 0
FIELDS = ["dataset", "reference", "reference_bits", "target_bytes_per_vector", "variant", "code_bits", "bytes_per_vector",
          "fits"]


def reference_targets(cells):
    """(codec, bits) -> effective bytes/vector, from each cell's first row (all rows of a cell share the build)."""
    targets = {}
    for path in cells:
        row = json.loads(Path(path).read_text())[0]
        targets[row["codec"], int(row["target_bits"])] = float(row["effective_bytes_per_vector"])
    return targets


class Probes:
    """Seed-0 bytes/vector per (variant, code_bits), shared across targets."""

    def __init__(self, vectors, log):
        self.vectors, self.log, self.measured = vectors, log, {}

    def __call__(self, variant, code_bits):
        key = variant, code_bits
        if key not in self.measured:
            dimensions = self.vectors.shape[1]
            bits = min(MAX_BITS_PER_DIM, -(-code_bits // dimensions))
            index = tieredq_index(self.vectors, bits, CALIBRATION_SEED, False, code_bits, variant)
            with tempfile.TemporaryDirectory(dir=self.log.parent) as directory:
                self.measured[key] = index.storage(str(Path(directory) / "index.bin")) / len(self.vectors)
        return self.measured[key]


def calibrate(probes, variant, target_bytes, dimensions):
    """Largest probed code_bits fitting target_bytes, and its bytes/vector; None if even 1 code bit overshoots."""
    target_bits = target_bytes * BITS_PER_BYTE
    fitting, failing = {}, {}
    for _ in range(MAX_PROBES):
        for (probed_variant, code), measured in probes.measured.items():
            if probed_variant == variant:
                (fitting if measured * BITS_PER_BYTE <= target_bits else failing)[code] = measured * BITS_PER_BYTE
        code_bits = next_probe(fitting, failing, target_bits, dimensions)
        if code_bits is None:
            break
        probes(variant, code_bits)
    if not fitting:
        return None
    best = max(fitting)
    return best, fitting[best] / BITS_PER_BYTE


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["openai", "gemini", "openai1m"])
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("cells", nargs="+", type=Path)
    args = parser.parse_args()
    import codec_native
    from data import prepare
    from threadpoolctl import threadpool_limits
    codec_native.set_threads(8)
    with threadpool_limits(limits=1):
        vectors, _, _ = prepare(args.data, args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / f"calibration-{args.dataset}.csv"
    probes = Probes(vectors, log)
    matches = []
    with log.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for (reference, bits), target in sorted(reference_targets(args.cells).items()):
            for variant in TIEREDQ_VARIANTS:
                seen = set(probes.measured)
                result = calibrate(probes, variant, target, vectors.shape[1])
                for (probed_variant, code), measured in probes.measured.items():
                    if (probed_variant, code) not in seen and probed_variant == variant:
                        writer.writerow(dict(dataset=args.dataset, reference=reference, reference_bits=bits,
                                             target_bytes_per_vector=target, variant=variant, code_bits=code,
                                             bytes_per_vector=measured, fits=measured <= target))
                handle.flush()
                if result is None:
                    print(f"{args.dataset} {reference} b={bits} {variant}: no fitting budget", flush=True)
                    continue
                code_bits, measured = result
                matches.append(dict(dataset=args.dataset, reference=reference, reference_bits=bits,
                                    target_bytes_per_vector=target, variant=variant, code_bits=code_bits,
                                    bytes_per_vector=measured))
                print(f"{args.dataset} {reference} b={bits} target={target:.2f} B {variant}: code_bits={code_bits} "
                      f"({measured:.2f} B)", flush=True)
    (args.output / f"matches-{args.dataset}.json").write_text(json.dumps(matches, indent=2) + "\n")


if __name__ == "__main__":
    main()
