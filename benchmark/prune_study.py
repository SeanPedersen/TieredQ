"""Feasibility study for two-stage tier-pruned code-lookup scans (no native kernel changes).

Stage 1 scores the widest tiers plus the exact metadata terms; the remaining tiers are bounded per query by
|weight| x codebook range. Only documents whose upper bound can still reach the top 10 are fully scored, so
survivor scores continue the sequential FP32 accumulator and are checked bit for bit. Reports pruned document and 16-document block fractions
per tier prefix, against the oracle top-10 threshold and a streaming per-segment threshold.
"""
import argparse
import itertools
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
from data import codec_vectors, prepare
from l2_transform import L2Transform
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
TOP_K = 10
FILE_HEADER_BYTES = 80
LEVEL_TABLES = 9
SCAN_BLOCK_DOCS = 16
STREAM_SEGMENTS = 8  # one contiguous document range per worker thread, each with its own heap
STREAM_CHUNK_DOCS = 256


class Reader:
    def __init__(self, data, pos=0):
        self.data, self.pos = data, pos

    def u32(self):
        self.pos += 4
        return struct.unpack_from("<I", self.data, self.pos - 4)[0]

    def u64(self):
        self.pos += 8
        return struct.unpack_from("<Q", self.data, self.pos - 8)[0]

    def floats(self):
        count = self.u64()
        values = np.frombuffer(self.data, dtype="<f4", count=count, offset=self.pos)
        self.pos += 4 * count
        return values


def load_exact_index(path):
    """Model arrays and packed rows from an exact-mode index file."""
    data = path.read_bytes()
    model_length, payload_offset, rows = struct.unpack_from("<Q", data, 24)[0], struct.unpack_from("<Q", data, 32)[0], \
        struct.unpack_from("<Q", data, 48)[0]
    reader = Reader(data, FILE_HEADER_BYTES)
    dimensions, code_bits, centroids = reader.u32(), reader.u32(), reader.u32()
    reader.u32(); reader.u32()  # rotation flag, model format
    code_bytes, stride, tier_count = reader.u32(), reader.u32(), reader.u32()
    tiers = [dict(zip(("start", "dimensions", "bits", "scale_group", "bit_offset"), (reader.u32() for _ in range(5))))
             for _ in range(tier_count)]
    model = {"dimensions": dimensions, "code_bits": code_bits, "code_bytes": code_bytes, "tiers": tiers}
    model["centroids"] = reader.floats().reshape(centroids, dimensions)
    basis, model["mean"], model["sigma"] = reader.floats(), reader.floats(), reader.floats()
    model["basis"] = basis.reshape(dimensions, len(model["sigma"]))
    if reader.u32() != LEVEL_TABLES:
        raise ValueError("unexpected level table count")
    model["levels"] = [reader.floats() for _ in range(LEVEL_TABLES)]
    if reader.pos != FILE_HEADER_BYTES + model_length:
        raise ValueError("model length mismatch")
    packed = np.frombuffer(data, dtype=np.uint8, offset=payload_offset).reshape(rows, stride)
    return model, packed


def unpack_codes(model, packed):
    """Code IDs per coded rank (documents x ranks)."""
    code_bytes = model["code_bytes"]
    padded = np.zeros((len(packed), code_bytes + 3), dtype=np.uint8)
    padded[:, :code_bytes] = packed[:, :code_bytes]
    codes = np.empty((len(packed), len(model["sigma"])), dtype=np.uint8)
    for tier in model["tiers"]:
        for j in range(tier["dimensions"]):
            offset = tier["bit_offset"] + j * tier["bits"]
            window = padded[:, offset // 8].astype(np.uint32) | padded[:, offset // 8 + 1].astype(np.uint32) << 8
            codes[:, tier["start"] + j] = (window >> (offset % 8)) & ((1 << tier["bits"]) - 1)
    return codes


def document_terms(model, packed):
    code_bytes = model["code_bytes"]
    centroid = packed[:, code_bytes].astype(np.intp)
    halves = np.ascontiguousarray(packed[:, code_bytes + 1:code_bytes + 5]).view("<f2").astype(np.float32)
    return centroid, halves[:, 0], halves[:, 1]


def query_terms(model, probes):
    """Mirror native FP32 operations in their original order, without fused multiply-add."""
    probes = np.asarray(probes, dtype=np.float32)
    projected = np.zeros((len(probes), len(model["sigma"])), dtype=np.float32)
    dots = np.zeros((len(probes), len(model["centroids"])), dtype=np.float32)
    for dimension in range(model["dimensions"]):
        projected += probes[:, dimension, None] * model["basis"][dimension]
        dots += probes[:, dimension, None] * model["centroids"][:, dimension]
    correction = np.zeros(len(probes), dtype=np.float32)
    for rank in range(projected.shape[1]):
        correction += projected[:, rank] * model["mean"][rank]
    norms = np.zeros(len(model["centroids"]), dtype=np.float64)
    for dimension in range(model["dimensions"]):
        values = model["centroids"][:, dimension].astype(np.float64)
        norms += values * values
    norms = (norms / 2).astype(np.float32) * np.float32(2)
    return (projected * model["sigma"], dots + correction[:, None],
            np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0))


def score_ranks(decoded, weights, ranks, initial=None, documents=None):
    """Sequential FP32 reference; continuing a prefix preserves full-scan rounding."""
    dot = np.zeros(decoded.shape[0], dtype=np.float32) if initial is None else initial.copy()
    for rank in ranks:
        levels = decoded[:, rank] if documents is None else decoded[documents, rank]
        dot += levels * weights[rank]
    return dot


def finish_scores(base, factor, dot):
    return base + factor * dot


def tier_prefixes(model):
    """Tier index lists, widest first; each prefix is scored exactly in stage 1."""
    order = sorted(range(len(model["tiers"])), key=lambda t: (-model["tiers"][t]["bits"], model["tiers"][t]["start"]))
    return [order[:size] for size in range(1, len(order))]


def tier_ranks(tier):
    return slice(tier["start"], tier["start"] + tier["dimensions"])


def streaming_thresholds(scores):
    """Per-document kth-best exact score seen so far in its segment, updated per chunk (-inf until k seen)."""
    count = len(scores)
    thresholds = np.full(count, -np.inf)
    bounds = np.linspace(0, count, STREAM_SEGMENTS + 1).astype(int)
    for begin, end in itertools.pairwise(bounds):
        best = np.empty(0)
        for chunk in range(begin, end, STREAM_CHUNK_DOCS):
            stop = min(chunk + STREAM_CHUNK_DOCS, end)
            if len(best) >= TOP_K:
                thresholds[chunk:stop] = best.min()
            merged = np.concatenate([best, scores[chunk:stop]])
            best = merged if len(merged) <= TOP_K else np.partition(merged, -TOP_K)[-TOP_K:]
    return thresholds


def block_survival(survivors):
    padded = np.zeros(-(-len(survivors) // SCAN_BLOCK_DOCS) * SCAN_BLOCK_DOCS, dtype=bool)
    padded[:len(survivors)] = survivors
    return float(padded.reshape(-1, SCAN_BLOCK_DOCS).any(axis=1).mean())


def remaining_ranks(tiers, prefix):
    return np.concatenate([np.arange(tier["start"], tier["start"] + tier["dimensions"])
                           for t, tier in enumerate(tiers) if t not in prefix])


def roundoff_allowance(model, weights, base, factor):
    """Conservative allowance for FP32 products, accumulation and score epilogue."""
    unit = np.finfo(np.float32).eps
    operations = 2 * len(weights) + 8
    gamma = operations * unit / (1 - operations * unit)
    magnitude = sum(np.sum(np.abs(weights[tier_ranks(tier)].astype(np.float64))) *
                    np.max(np.abs(model["levels"][tier["bits"]])) for tier in model["tiers"])
    return gamma * (np.abs(base.astype(np.float64)) + np.abs(factor) * magnitude)


def study(model, packed, probes, native_ids):
    codes = unpack_codes(model, packed)
    tiers = model["tiers"]
    decoded = np.empty(codes.shape, dtype=np.float32, order="F")
    for tier in tiers:
        decoded[:, tier_ranks(tier)] = model["levels"][tier["bits"]][codes[:, tier_ranks(tier)]]
    del codes
    centroid, factor, correction = document_terms(model, packed)
    weights, centroid_scores, correction_weights = query_terms(model, probes)
    prefixes = tier_prefixes(model)
    rests = [remaining_ranks(tiers, prefix) for prefix in prefixes]
    # Cauchy-Schwarz alternative: needs a stored per-document norm of the remaining decoded levels.
    rest_norms = [np.linalg.norm(decoded[:, rest].astype(np.float64), axis=1) for rest in rests]
    # Per prefix: summed pruned/survivor statistics over queries for each threshold and bound.
    stats = [{key: 0.0 for key in ("pruned_oracle", "pruned_stream", "blocks_oracle", "blocks_stream",
                                   "pruned_stream_cs", "blocks_stream_cs")} for _ in prefixes]
    top_overlap = 0.0
    verified_survivors = 0
    if any(prefix != list(range(len(prefix))) for prefix in prefixes):
        raise ValueError("widest-first tiers must be a coordinate prefix to continue exact FP32 sums")
    for row in range(len(probes)):
        base = centroid_scores[row][centroid] + correction_weights[row][centroid] * correction
        full_dot = score_ranks(decoded, weights[row], range(decoded.shape[1]))
        full = finish_scores(base, factor, full_dot)
        oracle = np.partition(full, -TOP_K)[-TOP_K]
        stream = streaming_thresholds(full)
        top_ids = np.argpartition(full, -TOP_K)[-TOP_K:]
        top_overlap += len(np.intersect1d(top_ids, native_ids[row])) / TOP_K
        allowance = roundoff_allowance(model, weights[row], base, factor)
        for index, prefix in enumerate(prefixes):
            rest = [t for t in range(len(tiers)) if t not in prefix]
            stop = sum(tiers[t]["dimensions"] for t in prefix)
            prefix_dot = score_ranks(decoded, weights[row], range(stop))
            partial = base.astype(np.float64) + factor.astype(np.float64) * prefix_dot
            low = high = 0.0
            for t in rest:
                book, wt = model["levels"][tiers[t]["bits"]], weights[row][tier_ranks(tiers[t])].astype(np.float64)
                high += np.sum(np.maximum(wt * book.max(), wt * book.min()))
                low += np.sum(np.minimum(wt * book.max(), wt * book.min()))
            upper = partial + np.where(factor >= 0, factor * high, factor * low) + allowance
            upper_cs = np.minimum(upper, partial + np.abs(factor) * np.linalg.norm(weights[row][rests[index]]) *
                                  rest_norms[index] + allowance)
            survive_oracle, survive_stream = upper >= oracle, upper >= stream
            survive_cs = upper_cs >= stream
            if not np.all(survive_cs[top_ids]) or not np.all(survive_stream[top_ids]):
                raise AssertionError("bound pruned a top-10 document")
            survivors = np.flatnonzero(survive_cs)
            continued = score_ranks(decoded, weights[row], range(stop, decoded.shape[1]),
                                    prefix_dot[survivors], documents=survivors)
            rescored = finish_scores(base[survivors], factor[survivors], continued)
            if not np.array_equal(rescored.view(np.uint32), full[survivors].view(np.uint32)):
                raise AssertionError("survivor scores differ from the sequential FP32 reference")
            verified_survivors += len(survivors)
            stats[index]["pruned_oracle"] += 1 - survive_oracle.mean()
            stats[index]["pruned_stream"] += 1 - survive_stream.mean()
            stats[index]["pruned_stream_cs"] += 1 - survive_cs.mean()
            stats[index]["blocks_oracle"] += block_survival(survive_oracle)
            stats[index]["blocks_stream"] += block_survival(survive_stream)
            stats[index]["blocks_stream_cs"] += block_survival(survive_cs)
    coded, code_bits = len(model["sigma"]), sum(t["dimensions"] * t["bits"] for t in tiers)
    results = []
    for prefix, entry in zip(prefixes, stats):
        averaged = {key: value / len(probes) for key, value in entry.items()}
        rank_fraction = sum(tiers[t]["dimensions"] for t in prefix) / coded
        bit_fraction = sum(tiers[t]["dimensions"] * tiers[t]["bits"] for t in prefix) / code_bits
        # Work model: stage 1 always scans the prefix; stage 2 scans the rest for surviving 16-document blocks.
        speedup = {f"speedup_{name}_{measure}": 1 / (fraction + averaged[f"blocks_{name}"] * (1 - fraction))
                   for name in ("oracle", "stream", "stream_cs")
                   for measure, fraction in (("ranks", rank_fraction), ("bits", bit_fraction))}
        results.append(dict(prefix_bits=[tiers[t]["bits"] for t in prefix],
                            prefix_ranks=sum(tiers[t]["dimensions"] for t in prefix),
                            rank_fraction=rank_fraction, bit_fraction=bit_fraction, **averaged, **speedup))
    return results, top_overlap / len(probes), verified_survivors


def build(args, dataset, bits):
    import codec_native
    base, queries, _ = prepare(args.data, dataset, None, args.queries)
    sift = dataset == "sift"
    transform = L2Transform.fit(base) if sift else None
    vectors, probes = (transform.base_vectors(base), transform.queries(queries)) if transform else \
        codec_vectors(base, queries, False)
    codec_native.set_threads(args.threads)
    index = codec_native.Index(vectors, bits, args.seed, True, sift, False)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "index.tq"
        index.storage(str(path))
        model, packed = load_exact_index(path)
    native_ids = np.empty((len(probes), TOP_K), dtype=np.uint32)
    index.search(np.ascontiguousarray(probes, dtype=np.float32), native_ids, args.threads)
    return model, np.array(packed), probes, native_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--native", type=Path, default=HERE / "build-adapters")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["sift", "glove", "openai"])
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh output file")
    sys.path.insert(0, str(args.native.resolve()))
    report = {"seed": args.seed, "queries": args.queries, "top_k": TOP_K, "stream_segments": STREAM_SEGMENTS,
                  "stream_chunk_docs": STREAM_CHUNK_DOCS, "scan_block_docs": SCAN_BLOCK_DOCS, "rows": []}
    with threadpool_limits(limits=args.threads):
        for dataset in args.datasets:
            for bits in args.bits:
                model, packed, probes, native_ids = build(args, dataset, bits)
                prefixes, overlap, verified = study(model, packed, probes, native_ids)
                tiers = [(t["bits"], t["dimensions"]) for t in model["tiers"]]
                report["rows"].append({"dataset": dataset, "bits": bits, "tiers": tiers, "top_overlap": overlap,
                                           "bit_exact_survivor_checks": verified, "prefixes": prefixes})
                print(f"{dataset} {bits}-bit tiers={tiers} top-10 overlap with native search {overlap:.4f}", flush=True)
                for entry in prefixes:
                    print(f"  prefix {entry['prefix_bits']}: ranks {entry['rank_fraction']:.2f} "
                          f"pruned oracle {entry['pruned_oracle']:.3f} stream {entry['pruned_stream']:.3f} "
                          f"cs {entry['pruned_stream_cs']:.3f} | blocks stream {entry['blocks_stream']:.3f} "
                          f"| speedup stream {entry['speedup_stream_ranks']:.2f}x cs {entry['speedup_stream_cs_ranks']:.2f}x",
                          flush=True)
                args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
