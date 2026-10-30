"""DBpedia OpenAI-1536 at 1M: the `openai` 100K set's 1,000 held-out queries against all 999,000 other rows.

The query rows come from the same seed-42 draws as data.py's `openai` sample, so both sets share their queries
bit for bit. The float64 embedding column is streamed chunk by chunk from the Arrow-backed dataset straight into
normalized float32 base and query arrays; `datasets.select` over 1M lists is too slow and memory-hungry.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

DIMENSIONS = 1536
ROWS = 1_000_000
BASE_ROWS = ROWS - 1000
SAMPLED_ROWS = 101_000


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def query_row_ids(rows, query_count):
    """Source rows of the `openai` 100K set's held-out queries (data.py draws, same order)."""
    from data import QUERY_COUNT, SPLIT_SEED
    rng = np.random.default_rng(SPLIT_SEED)
    indices = np.sort(rng.choice(rows, SAMPLED_ROWS, replace=False))
    positions = np.sort(rng.choice(len(indices), QUERY_COUNT, replace=False))
    return indices[positions], indices[positions[:query_count]]


def stream_split(column, held_out, selected_queries, base_n):
    """Normalized float32 base (rows not held out, source order, first base_n) and selected queries."""
    from data import normalize
    base = np.empty((base_n, DIMENSIONS), dtype=np.float32)
    queries = np.empty((len(selected_queries), DIMENSIONS), dtype=np.float32)
    is_held_out = np.zeros(column.length(), dtype=bool)
    is_held_out[held_out] = True
    query_slot = {int(row): slot for slot, row in enumerate(selected_queries)}
    start = filled = found = 0
    for chunk in column.chunks:
        stop = start + len(chunk)
        if np.any(np.diff(chunk.offsets.to_numpy()) != DIMENSIONS):
            raise ValueError(f"Ragged or unexpected embedding dimensions in rows {start}-{stop}")
        rows = normalize(chunk.flatten().to_numpy().astype(np.float32).reshape(len(chunk), DIMENSIONS))
        local = np.flatnonzero(is_held_out[start:stop])
        for position in local:
            if (slot := query_slot.get(start + int(position))) is not None:
                queries[slot] = rows[position]
                found += 1
        kept = np.delete(rows, local, axis=0)[:base_n - filled]
        base[filled:filled + len(kept)] = kept
        filled, start = filled + len(kept), stop
        if filled == base_n and found == len(query_slot):
            break
    if filled != base_n or found != len(query_slot):
        raise ValueError(f"Filled {filled} of {base_n} base rows and {found} of {len(query_slot)} queries")
    return base, queries


def prepare_openai1m(root, max_base=None, query_count=1000):
    from data import SOURCES, ground_truth
    from datasets import load_dataset
    root = Path(root)
    n = min(max_base or BASE_ROWS, BASE_ROWS)
    cache = root / f"openai1m-{n}-{query_count}-v2.npz"
    if cache.exists():
        with np.load(cache) as data:
            return data["base"], data["queries"], data["truth"]
    source = SOURCES["openai"]
    dataset = load_dataset(source["repo"], split="train", revision=source["revision"])
    column = dataset.data.table.column(source["column"])
    if column.length() != ROWS:
        raise ValueError(f"Unexpected dataset: {column.length()} rows")
    held_out, selected_queries = query_row_ids(ROWS, query_count)
    base, queries = stream_split(column, held_out, selected_queries, n)
    if not (np.isfinite(base).all() and np.isfinite(queries).all()):
        raise ValueError("Nonfinite vectors")
    print(f"Exact FP32 truth: openai1m {base.shape}, queries={len(queries)}", flush=True)
    truth = ground_truth(base, queries, False)
    np.save(root / "openai1m-query-row-ids.npy", selected_queries)
    np.savez(cache, base=base, queries=queries, truth=truth)
    manifest = dict(source=source, dataset="openai1m", rows=ROWS, base_n=n, queries=query_count,
                    split="data.py openai seed-42 held-out queries; base = every other row in source order",
                    normalization="FP32 L2 normalization for both base and queries",
                    truth="exhaustive chunked FP32 dot product top-10",
                    base_sha256=digest(base), queries_sha256=digest(queries), truth_sha256=digest(truth),
                    query_row_ids_sha256=digest(selected_queries))
    cache.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return base, queries, truth
