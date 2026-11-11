"""Compare fast and FP32 TieredQ recall using identical training and full datasets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from threadpoolctl import threadpool_limits

from data import codec_vectors, prepare
from measure import L2Search, measure, recall

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=HERE / "build-adapters")
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", nargs="+", type=int, default=[8])
    parser.add_argument("--datasets", nargs="+", default=["sift", "glove", "openai"])
    parser.add_argument("--modes", nargs="+", choices=["fast", "fp32"], default=["fast", "fp32"])
    parser.add_argument("--lift-scale", type=float, default=1)
    parser.add_argument("--recall-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query-metric", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh output file")
    sys.path.insert(0, str(args.native.resolve()))
    import codec_native

    result = {"affinity": sorted(os.sched_getaffinity(0)), "seed": args.seed, "rows": [],
              "source_sha256": {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for folder in (HERE.parent / "src", HERE.parent / "include", HERE / "codec_adapters")
                                for p in folder.rglob("*") if p.is_file()}}
    with threadpool_limits(limits=1):
        for dataset in args.datasets:
            base, queries, truth = prepare(args.data, dataset, None, 1000)
            vectors, probes = codec_vectors(base, queries, dataset == "sift")
            if dataset == "sift":
                vectors[:, -1] *= args.lift_scale
                probes[:, -1] /= args.lift_scale
                if args.query_metric:
                    sample = np.column_stack([vectors[:32768, :-1], np.full(min(32768, len(vectors)), 1 / args.lift_scale)]).astype(np.float64)
                    moment = sample.T @ sample / len(sample)
                    moment += np.eye(moment.shape[0]) * np.finfo(np.float64).eps * np.trace(moment)
                    transform = np.linalg.cholesky(moment)
                    vectors = np.ascontiguousarray(vectors @ transform.astype(np.float32))
                    probes = np.ascontiguousarray(probes @ np.linalg.inv(transform).T.astype(np.float32))
            for bits in args.bits:
                for mode in args.modes:
                    fast = mode == "fast"
                    codec_native.set_threads(8)
                    index = codec_native.Index(vectors, bits, args.seed, True, dataset == "sift", fast)
                    layout = index.diagnostics()
                    with tempfile.TemporaryDirectory() as temporary:
                        storage = index.storage(str(Path(temporary) / "index"))
                    timed_lift = dataset == "sift" and args.lift_scale == 1 and not args.query_metric
                    search = L2Search(index, float(np.max(np.linalg.norm(base, axis=1)))) if timed_lift else index
                    if args.recall_only:
                        ids = np.empty((len(probes), 10), dtype=np.uint32)
                        for start in range(0, len(probes), 32):
                            search.search((queries if timed_lift else probes)[start:start+32], ids[start:start+32], 8)
                        metrics = {"recall10": recall(truth, ids)}
                    else:
                        metrics = measure(search, queries if timed_lift else probes, truth, 8, 32)
                    row = dict(dataset=dataset, bits=bits, fast=fast, layout=layout,
                               lift_scale=args.lift_scale, timed_lift=timed_lift,
                               query_metric=args.query_metric,
                               bytes_per_vector=storage / len(base), **metrics)
                    result["rows"].append(row)
                    args.output.write_text(json.dumps(result, indent=2))
                    print(json.dumps(row), flush=True)
                    del search, index


if __name__ == "__main__":
    main()
