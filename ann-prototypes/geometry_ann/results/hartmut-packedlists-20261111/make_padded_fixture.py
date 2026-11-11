"""Writes tests/fixtures/tieredq_ivf_padded.*: IVF indexes saved by a build before #53 (padded lists, payload versions
4 and 5) with the rows they were built from, queries, and the ids and scores they return. The C++ test loads each
file, rebuilds it in place from the rows (the loaded lists stay fixed, so the partition and codes are the same) and
requires the packed rebuild to return the same ids and scores.

Usage: python make_padded_fixture.py OUT_DIR   (with the tieredq package of the pre-#53 build)"""
import sys
from pathlib import Path

import numpy as np
from tieredq import TieredQIVFIndex

ROWS, DIMENSIONS, CENTERS, QUERIES, LISTS, K, NPROBE = 1200, 32, 12, 16, 40, 10, 6
MODELS = {"ip2": dict(metric="ip", bits=2, router=True), "l2x6": dict(metric="l2", bits=6, router=False)}


def clustered(rng, count, centers):
    return (centers[rng.integers(0, len(centers), count)] + 0.6 * rng.standard_normal((count, DIMENSIONS))).astype(np.float32)


def main():
    out = Path(sys.argv[1])
    rng = np.random.default_rng(53)
    centers = 3.0 * rng.standard_normal((CENTERS, DIMENSIONS))
    rows, queries = clustered(rng, ROWS, centers), clustered(rng, QUERIES, centers)
    rows.tofile(out / "tieredq_ivf_padded.rows")
    queries.tofile(out / "tieredq_ivf_padded.queries")
    for name, model in MODELS.items():
        index = TieredQIVFIndex(dimensions=DIMENSIONS, code_bits=model["bits"] * DIMENSIONS, lists=LISTS, metric=model["metric"],
                                router=False)
        index.build(rows)
        if model["router"]:
            index.build_router(lists_per_super=8, rerank=16, min_super_probes=2)
        index.save(str(out / f"tieredq_ivf_padded_{name}.tq"))
        scores, ids = index.search(queries, k=K, nprobe=NPROBE)
        ids.astype("<u4").tofile(out / f"tieredq_ivf_padded_{name}.ids")
        scores.astype("<f4").tofile(out / f"tieredq_ivf_padded_{name}.scores")
        print(name, index.lists, index.has_router, index.storage_bytes)


if __name__ == "__main__":
    main()
