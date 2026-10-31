"""MS MARCO passages at 1M, Cohere embed-english-v3.0 (1,024-d), with two query sets on one base (#22 phase 1).

- `real`: 1,000 MS MARCO dev questions (embedded as search_query), seed 42 from the 6,980 dev queries.
- `doc`: 1,000 held-out corpus passages (search_document), the DBpedia setup.
The base is 1,000,000 passages drawn with seed 42 from the first two corpus shards (about 1.96M rows, ~4.2 GB instead
of the full 8.8M rows); the doc queries come from the same draw and are disjoint from it. Their recall difference
isolates the query distribution with everything else fixed. Rows are normalized FP32; truth is exact inner-product
top-10 and top-100. Shards stream batch by batch, so peak memory stays near the base itself.
A third set, `train`, is a seed-42 sample of MS MARCO train questions (disjoint from dev) for fitting query-aware
allocation; it is cached on its own and never enters the v1 base cache or its truth.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

DIMENSIONS = 1024
BASE_ROWS = 1_000_000
QUERY_COUNT = 1000
DEEP_K = 100
CORPUS_SHARDS = ("msmarco/corpus/0000.parquet", "msmarco/corpus/0001.parquet")
DEV_QUERIES = "msmarco/queries/dev.parquet"
TRAIN_QUERIES = "msmarco/queries/train.parquet"
TRAIN_SAMPLE = 20_000
QUERY_SETS = ("real", "doc")
BATCH_ROWS = 65_536
TRUTH_BASE_CHUNK = 65_536


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def fetch(root, path):
    """The pinned file under root, downloaded once and checked against its LFS sha256."""
    from data import SOURCES, download
    source = SOURCES["msmarco"]
    target = root / "msmarco" / Path(path).name
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{path}"
    download(url, target, source["files"][path])
    return target


def embeddings(batch):
    """A record batch's `emb` column as float32 rows."""
    column = batch.column("emb")
    if np.any(np.diff(column.offsets.to_numpy()) != DIMENSIONS):
        raise ValueError("Ragged or unexpected embedding dimensions")
    return column.values.to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(batch), DIMENSIONS)


def split_rows(total, base_n):
    """Seed-42 draw of base_n + QUERY_COUNT source rows; the doc queries are QUERY_COUNT positions of that draw."""
    from data import SPLIT_SEED
    rng = np.random.default_rng(SPLIT_SEED)
    drawn = np.sort(rng.choice(total, BASE_ROWS + QUERY_COUNT, replace=False))
    positions = np.sort(rng.choice(len(drawn), QUERY_COUNT, replace=False))
    is_query = np.zeros(len(drawn), dtype=bool)
    is_query[positions] = True
    return drawn[~is_query][:base_n], drawn[is_query]


def stream_rows(files, wanted):
    """Normalized rows and ids at the sorted global row numbers `wanted`, in that order."""
    import pyarrow.parquet as pq
    from data import normalize
    rows = np.empty((len(wanted), DIMENSIONS), dtype=np.float32)
    ids = []
    start = filled = 0
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=BATCH_ROWS, columns=["_id", "emb"]):
            stop = start + len(batch)
            local = wanted[filled:np.searchsorted(wanted, stop)] - start
            if len(local):
                rows[filled:filled + len(local)] = normalize(embeddings(batch.take(local)))
                ids.extend(batch.column("_id").take(local).to_pylist())
                filled += len(local)
            start = stop
    if filled != len(wanted):
        raise ValueError(f"Found {filled} of {len(wanted)} rows")
    return rows, np.asarray(ids)


def exact_top(base, queries, k):
    """Exact inner-product top-k ids per query (ties to the lower id)."""
    best_scores = np.full((len(queries), k), -np.inf, dtype=np.float32)
    best_ids = np.zeros((len(queries), k), dtype=np.uint32)
    for offset in range(0, len(base), TRUTH_BASE_CHUNK):
        scores = queries @ base[offset:offset + TRUTH_BASE_CHUNK].T
        local = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        scores = np.concatenate([best_scores, np.take_along_axis(scores, local, axis=1)], axis=1)
        ids = np.concatenate([best_ids, (local + offset).astype(np.uint32)], axis=1)
        top = np.lexsort((ids, -scores), axis=1)[:, :k]
        best_scores, best_ids = np.take_along_axis(scores, top, axis=1), np.take_along_axis(ids, top, axis=1)
    return best_ids


def build(root, base_n):
    import pyarrow.parquet as pq
    from data import SOURCES, SPLIT_SEED
    files = [fetch(root, path) for path in CORPUS_SHARDS]
    total = sum(pq.ParquetFile(file).metadata.num_rows for file in files)
    base_rows, doc_rows = split_rows(total, base_n)
    base, base_ids = stream_rows(files, base_rows)
    doc, doc_ids = stream_rows(files, doc_rows)
    dev = fetch(root, DEV_QUERIES)
    dev_count = pq.ParquetFile(dev).metadata.num_rows
    real_rows = np.sort(np.random.default_rng(SPLIT_SEED).choice(dev_count, QUERY_COUNT, replace=False))
    real, real_ids = stream_rows([dev], real_rows)
    arrays = dict(base=base, base_ids=base_ids, real=real, real_ids=real_ids, doc=doc, doc_ids=doc_ids)
    for name, queries in (("real", real), ("doc", doc)):
        print(f"Exact FP32 truth: msmarco1m {name} {base.shape}, queries={len(queries)}", flush=True)
        arrays[f"truth_{name}"] = exact_top(base, queries, DEEP_K)
    manifest = dict(source=SOURCES["msmarco"], dataset="msmarco1m", corpus_rows=int(total), base_n=base_n,
                    queries=QUERY_COUNT, dev_queries=int(dev_count),
                    split="seed 42: base + doc queries drawn from the first two corpus shards; real = dev queries",
                    normalization="FP32 L2 normalization for base and both query sets",
                    truth=f"exhaustive FP32 inner-product top-{DEEP_K} (top-10 = first 10)",
                    **{f"{key}_sha256": digest(value) for key, value in arrays.items() if value.dtype != object})
    return arrays, manifest


def load_msmarco1m(root, max_base=None):
    """All arrays of the msmarco1m cache (built on first use)."""
    root = Path(root)
    n = min(max_base or BASE_ROWS, BASE_ROWS)
    cache = root / f"msmarco1m-{n}-{QUERY_COUNT}-v1.npz"
    if not cache.exists():
        arrays, manifest = build(root, n)
        np.savez(cache, **arrays)
        cache.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    with np.load(cache) as data:
        return {key: data[key] for key in data.files}


def prepare_msmarco1m(root, query_set, max_base=None, query_count=QUERY_COUNT):
    """data.prepare's (base, queries, top-10 truth) for one query set."""
    if query_set not in QUERY_SETS:
        raise ValueError(f"Unknown msmarco1m query set {query_set}")
    arrays = load_msmarco1m(root, max_base)
    return arrays["base"], arrays[query_set][:query_count], arrays[f"truth_{query_set}"][:query_count, :10]


def load_train_queries(root, count=TRAIN_SAMPLE):
    """Seed-42 sample of `count` normalized train questions with their ids (cached as msmarco-train-queries-*-v1)."""
    import pyarrow.parquet as pq
    from data import SPLIT_SEED
    root = Path(root)
    cache = root / f"msmarco-train-queries-{count}-v1.npz"
    if not cache.exists():
        source = fetch(root, TRAIN_QUERIES)
        total = pq.ParquetFile(source).metadata.num_rows
        rows = np.sort(np.random.default_rng(SPLIT_SEED).choice(total, count, replace=False))
        queries, ids = stream_rows([source], rows)
        np.savez(cache, queries=queries, ids=ids)
        cache.with_suffix(".json").write_text(json.dumps(dict(
            file=TRAIN_QUERIES, train_queries=int(total), count=count, split=f"seed {SPLIT_SEED} sample of train rows",
            queries_sha256=digest(queries)), indent=2) + "\n")
    with np.load(cache) as data:
        return data["queries"], data["ids"]


def export(root, output):
    """Eval directories output/msmarco1m-{real,doc} (base.f32, queries.f32, truth.u32 top-10, truth100.u32); the doc
    set's base.f32 links to the real set's."""
    arrays = load_msmarco1m(root)
    first = None
    for name in QUERY_SETS:
        target = Path(output) / f"msmarco1m-{name}"
        target.mkdir(parents=True, exist_ok=True)
        if first is None:
            arrays["base"].tofile(target / "base.f32")
            first = target / "base.f32"
        elif not (target / "base.f32").exists():
            (target / "base.f32").symlink_to(first.resolve())
        arrays[name].tofile(target / "queries.f32")
        np.ascontiguousarray(arrays[f"truth_{name}"][:, :10]).tofile(target / "truth.u32")
        arrays[f"truth_{name}"].tofile(target / "truth100.u32")
        print(name, arrays["base"].shape, arrays[name].shape, flush=True)


def export_train_queries(root, output):
    """output/train_queries.f32: the train-question sample, for QUERY_SAMPLE in the IVF eval."""
    queries, _ = load_train_queries(root)
    Path(output).mkdir(parents=True, exist_ok=True)
    queries.tofile(Path(output) / "train_queries.f32")
    print("train", queries.shape, flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "train-queries":
        export_train_queries(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        export(Path(sys.argv[1]), Path(sys.argv[2]))
