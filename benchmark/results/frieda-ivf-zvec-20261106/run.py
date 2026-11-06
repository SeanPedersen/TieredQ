"""Repeat the existing IVF comparison, preserving per-pass timings and median-QPS TSVs."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

DEFAULT_REPEATS = 3
DIMENSIONS = 768
LISTS = 256
BITS_PER_BYTE = 8
NPROBES = (4, 8, 16, 32, 64, 128, 256)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--codec", choices=("zvec", "tieredq", "nested"), required=True)
    parser.add_argument("--bits", type=float, required=True)
    parser.add_argument("--prefix", type=float, default=1)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    return args


def make_index(args, harness):
    if args.codec == "tieredq":
        return "tieredq", harness.TieredQIVF(DIMENSIONS, round(args.bits * DIMENSIONS), LISTS, 0)
    if args.codec == "nested":
        label = f"nested-p{args.prefix:g}-c{args.candidates}"
        return label, harness.TieredQNested(
            DIMENSIONS, round(args.bits * DIMENSIONS), LISTS,
            round(args.prefix * DIMENSIONS), args.candidates)
    path = args.workdir / f"zvec-b{int(args.bits)}-l{LISTS}"
    if path.exists():
        raise FileExistsError(f"Refusing to replace an existing zvec index: {path}")
    args.workdir.mkdir(parents=True, exist_ok=True)
    return "zvec", harness.ZvecIVF(args.workdir, DIMENSIONS, int(args.bits), LISTS)


def measure(args, harness):
    base, queries, truth = harness.load(args.data, DIMENSIONS)
    label, index = make_index(args, harness)
    print(f"BUILD {label} bits={args.bits:g}", flush=True)
    start = time.perf_counter()
    index.build(base)
    build_s = time.perf_counter() - start
    index_bits = index.index_bytes() * BITS_PER_BYTE / len(base)
    output = args.output / f"{label}.tsv"
    samples = args.output / f"{label}.jsonl"
    args.output.mkdir(parents=True, exist_ok=True)
    with output.open("a") as table, samples.open("a") as details:
        for nprobe in NPROBES:
            search = index.searcher(queries, nprobe)
            passes = []
            for _ in range(args.repeats):
                ids, qps = harness.timed_qps(search, len(queries))
                passes.append(dict(qps=qps, recall=float(harness.recall(ids, truth))))
            recalls = {row["recall"] for row in passes}
            if len(recalls) != 1:
                raise RuntimeError(f"Recall changed between timing passes: {passes}")
            median_qps = statistics.median(row["qps"] for row in passes)
            recall = passes[0]["recall"]
            row = (f"{label}\t{args.bits:g}\t{LISTS}\t{nprobe}\t{index_bits:.0f}"
                   f"\t{recall:.4f}\t{median_qps:.1f}\t{build_s:.1f}")
            table.write(row + "\n")
            table.flush()
            details.write(json.dumps(dict(
                codec=label, bits=args.bits, lists=LISTS, nprobe=nprobe,
                index_bits_per_vector=index_bits, build_s=build_s,
                queries=len(queries), passes=passes)) + "\n")
            details.flush()
            print(row, flush=True)


def main():
    args = arguments()
    sys.path.insert(0, str(args.harness.resolve()))
    import ivf_vs_zvec
    measure(args, ivf_vs_zvec)


if __name__ == "__main__":
    main()
