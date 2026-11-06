# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py>=3.10", "numpy>=1.26"]
# ///
"""Export full ANN-Benchmarks base plus disjoint calibration/test query sets and known truth."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from vecs_io import write_records

HASH_BLOCK_BYTES = 1 << 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--calibration", type=int, default=250)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if min(args.calibration, args.test) <= 0:
        raise ValueError("Query counts must be positive")
    digest = hashlib.sha256()
    with args.source.open("rb") as source:
        while block := source.read(HASH_BLOCK_BYTES):
            digest.update(block)
    checksum = digest.hexdigest()
    if args.expected_sha256 and checksum != args.expected_sha256:
        raise ValueError(f"Source SHA256 mismatch: {checksum}")
    args.output.mkdir(parents=True, exist_ok=True)
    names = ("base.fvecs", "calibration_queries.fvecs", "calibration_truth.ivecs", "test_queries.fvecs", "test_truth.ivecs", "manifest.json")
    if any((args.output / name).exists() for name in names):
        raise FileExistsError("Output already contains benchmark artifacts")
    with h5py.File(args.source, "r") as source:
        if len(source["train"]) < 1_000_000:
            raise ValueError("Experiment requires at least one million base vectors")
        if args.calibration + args.test > len(source["test"]):
            raise ValueError("Insufficient independent queries")
        base = np.asarray(source["train"], dtype=np.float32)
        queries = np.asarray(source["test"], dtype=np.float32)
        truth = np.asarray(source["neighbors"], dtype=np.int32)
    if not np.isfinite(base).all() or not np.isfinite(queries).all():
        raise ValueError("Nonfinite coordinates")
    order = np.random.default_rng(args.seed).permutation(len(queries))
    calibration = order[: args.calibration]
    test = order[args.calibration : args.calibration + args.test]
    print(f"Full base={base.shape}; calibration={len(calibration)}, held-out test={len(test)}", flush=True)
    hashes = {names[0]: write_records(args.output / names[0], base, "<f4")}
    for prefix, ids in (("calibration", calibration), ("test", test)):
        for suffix, data, dtype in (("queries.fvecs", queries[ids], "<f4"), ("truth.ivecs", truth[ids], "<i4")):
            name = f"{prefix}_{suffix}"
            hashes[name] = write_records(args.output / name, data, dtype)
            print(f"Wrote {name}", flush=True)
    manifest = dict(source=str(args.source), source_sha256=checksum, base_shape=base.shape, metric="squared L2",
                    normalization="none", seed=args.seed, calibration_query_ids=calibration.tolist(),
                    test_query_ids=test.tolist(), ground_truth="ANN-Benchmarks supplied exact neighbors", sha256=hashes)
    with (args.output / names[-1]).open("x") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
