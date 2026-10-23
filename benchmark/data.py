"""Deterministic dataset splits, normalization and cached exact FP32 truth."""
import hashlib
import json
from pathlib import Path
import subprocess
import h5py
import numpy as np

TOP_K = 10
QUERY_COUNT = 1000
BASE_COUNTS = {"sift": 1_000_000, "glove": 100_000, "openai": 100_000}
DIMENSIONS = {"sift": 128, "glove": 200, "openai": 1536}
SPLIT_SEED = 42
SOURCES = json.loads(Path(__file__).with_name("datasets.json").read_text())
HASH_CHUNK_BYTES = 1024 * 1024

def download(url, target, expected_sha256):
    if target.exists():
        verify_checksum(target, expected_sha256)
        return
    partial = target.with_suffix(".partial")
    print(f"Downloading {url}", flush=True)
    subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--output", str(partial), url], check=True)
    verify_checksum(partial, expected_sha256)
    partial.replace(target)


def verify_checksum(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Dataset checksum mismatch: {path}")


def normalize(rows):
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(rows).all():
        raise ValueError("Dataset contains zero or nonfinite vectors")
    return np.ascontiguousarray(rows / norms, dtype=np.float32)


def prepare(root, name, max_base=None, query_count=QUERY_COUNT):
    root.mkdir(parents=True, exist_ok=True)
    n = min(max_base or BASE_COUNTS[name], BASE_COUNTS[name])
    if n < TOP_K or not 1 <= query_count <= QUERY_COUNT:
        raise ValueError("Need at least 10 base rows and 1–1000 queries")
    cache = root / f"{name}-{n}-{query_count}-v2.npz"
    if cache.exists():
        with np.load(cache) as data:
            return data["base"], data["queries"], data["truth"]
    rng = np.random.default_rng(SPLIT_SEED)
    if name in {"sift", "glove"}:
        source = root / f"{name}.hdf5"
        download(SOURCES[name]["url"], source, SOURCES[name]["sha256"])
        with h5py.File(source) as handle:
            # GloVe uses the reference repo's sorted 100K sample, even for smoke runs.
            indices = (np.arange(BASE_COUNTS[name]) if name == "sift" else
                       np.sort(rng.choice(len(handle["train"]), BASE_COUNTS[name], replace=False)))
            base = (np.asarray(handle["train"][:n], dtype=np.float32) if name == "sift" else
                    np.asarray(handle["train"], dtype=np.float32)[indices[:n]])
            queries = np.asarray(handle["test"][:query_count], dtype=np.float32)
        np.save(root / f"{name}-base-indices.npy", indices)
    else:
        from datasets import load_dataset
        dataset = load_dataset(SOURCES[name]["repo"], split="train", revision=SOURCES[name]["revision"])
        # Match the reference sample and disjoint test positions before truncation.
        indices = np.sort(rng.choice(len(dataset), BASE_COUNTS[name] + QUERY_COUNT, replace=False))
        positions = np.sort(rng.choice(len(indices), QUERY_COUNT, replace=False))
        mask = np.ones(len(indices), dtype=bool)
        mask[positions] = False
        column = SOURCES[name]["column"]
        base = np.asarray(dataset.select(indices[mask][:n].tolist())[column], dtype=np.float32)
        queries = np.asarray(dataset.select(indices[positions[:query_count]].tolist())[column], dtype=np.float32)
        np.save(root / "openai-sampled-indices.npy", indices)
        np.save(root / "openai-test-positions.npy", positions)
    if name != "sift":
        base, queries = normalize(base), normalize(queries)
    if base.shape != (n, DIMENSIONS[name]) or queries.shape != (query_count, DIMENSIONS[name]):
        raise ValueError("Unexpected dataset dimensions")
    print(f"Exact FP32 truth: {name} {base.shape}, queries={len(queries)}", flush=True)
    truth = ground_truth(base, queries, name == "sift")
    np.savez(cache, base=base, queries=queries, truth=truth)
    manifest = {"source": SOURCES[name], "dataset": name, "split_seed": SPLIT_SEED, "base_n": n, "queries": query_count,
                "base_sha256": hashlib.sha256(base.tobytes()).hexdigest(),
                "queries_sha256": hashlib.sha256(queries.tobytes()).hexdigest()}
    cache.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    return base, queries, truth


def ground_truth(base, queries, l2):
    query_chunk, base_chunk = 32, 16384
    output = []
    for start in range(0, len(queries), query_chunk):
        q = queries[start:start + query_chunk]
        best_scores = np.full((len(q), TOP_K), -np.inf, dtype=np.float32)
        best_ids = np.zeros((len(q), TOP_K), dtype=np.uint32)
        for offset in range(0, len(base), base_chunk):
            block = base[offset:offset + base_chunk]
            scores = q @ block.T
            if l2:
                scores = 2 * scores - np.sum(block * block, axis=1)
            local_k = min(TOP_K, len(block))
            local = np.argpartition(-scores, local_k - 1, axis=1)[:, :local_k]
            scores = np.concatenate([best_scores, np.take_along_axis(scores, local, axis=1)], axis=1)
            ids = np.concatenate([best_ids, (local + offset).astype(np.uint32)], axis=1)
            top = np.argsort(-scores, axis=1, kind="stable")[:, :TOP_K]
            best_scores = np.take_along_axis(scores, top, axis=1)
            best_ids = np.take_along_axis(ids, top, axis=1)
        output.append(best_ids)
    return np.concatenate(output)


def codec_vectors(base, queries, l2):
    if not l2:
        return base, queries
    # Common scaling preserves L2 rankings and avoids a poorly conditioned IP lift.
    scale = float(np.max(np.linalg.norm(base, axis=1)))
    x, q = base / scale, queries / scale
    return (np.ascontiguousarray(np.column_stack([x, -np.sum(x*x, axis=1)/2]), dtype=np.float32),
            np.ascontiguousarray(np.column_stack([q, np.ones(len(q))]), dtype=np.float32))
