"""#33 measurement: bits/vector a lossless compact format could save over the fast index, per (dataset, budget, variant).

Usage: run.py OUTPUT_CSV DATASET=DIR:DIMENSIONS... [--bits 1,2,4,8]
Each DIR holds base.f32. Prints one line per cell and appends it to OUTPUT_CSV.
"""
import argparse
import csv
from pathlib import Path
import time

import numpy as np

import compact_native
from entropy import code_bits, elias_fano_bits, fine_split_bits, half_bits, kt_bits, metadata_fields

CENTROIDS = 256
# (fine_tiers, exact_fast_scores) of the public index variants
VARIANTS = {"fine": (True, False), "nofine": (False, False), "exact": (True, True)}
EXACT_BITS = 8  # exact_fast_scores is the 8 bits/dim batch-throughput option (#31)
WHOLE_CODE = 32  # prefix width of models without fine tiers: wider than any tier, so nothing is fine


def measure(base, bits_per_dim, variant):
    count, dimensions = base.shape
    fine, exact = VARIANTS[variant]
    started = time.perf_counter()
    model = compact_native.train_encode(base, bits_per_dim * dimensions, "ip", fine, exact)
    rows, tiers = np.asarray(model["rows"]), np.asarray(model["tiers"])
    centroids, factors, corrections = metadata_fields(rows)
    list_sizes = np.bincount(centroids, minlength=CENTROIDS)
    per_vector = lambda bits: bits / count
    return dict(
        bits_per_dim=bits_per_dim, variant=variant, rows=count, code_bits=model["code_bits"],
        prefix_bits=model["prefix_bits"],
        fast_ram_bits=per_vector(8 * model["fast_storage_bytes"]),
        fast_stride_bits=8 * model["fast_bytes_per_vector"],
        model_bits=per_vector(8 * model["model_bytes"]),
        codes_order0=per_vector(code_bits(rows, tiers)),
        codes_by_centroid=per_vector(code_bits(rows, tiers, centroids, CENTROIDS)),
        centroid=per_vector(kt_bits(list_sizes, CENTROIDS)),
        factor=per_vector(half_bits(factors)), correction=per_vector(half_bits(corrections)),
        ivf_ids=per_vector(elias_fano_bits(list_sizes, count)),
        **{key: per_vector(bits) for key, bits in fine_split_bits(rows, tiers, model["prefix_bits"] or WHOLE_CODE).items()},
        seconds=time.perf_counter() - started)


def cells(bits, variants):
    for bits_per_dim in bits:
        for variant in variants:
            if variant != "exact" or bits_per_dim == EXACT_BITS:
                yield bits_per_dim, variant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("datasets", nargs="+", help="NAME=DIR:DIMENSIONS")
    parser.add_argument("--bits", default="1,2,4,8")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    args = parser.parse_args()
    bits = [int(b) for b in args.bits.split(",")]
    writer = None
    with args.output.open("a", newline="") as handle:
        for spec in args.datasets:
            name, location = spec.split("=")
            directory, dimensions = location.rsplit(":", 1)
            base = np.fromfile(Path(directory) / "base.f32", np.float32).reshape(-1, int(dimensions))
            for bits_per_dim, variant in cells(bits, args.variants.split(",")):
                row = dict(dataset=name) | measure(base, bits_per_dim, variant)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    if handle.tell() == 0:
                        writer.writeheader()
                writer.writerow(row)
                handle.flush()
                print(" ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                      flush=True)


if __name__ == "__main__":
    main()
