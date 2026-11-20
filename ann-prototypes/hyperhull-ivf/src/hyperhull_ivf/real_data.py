"""Loaders for real embedding benchmarks stored under ~/data/ann.

  dbpedia: 305,696 x 1536 float16 unit-norm text embeddings (OpenAI-style), 2000 queries
  glove:   1,183,514 x 100 GloVe word vectors (angular -> L2-normalized here), 2000 queries
Base sets are returned as memory maps (or normalized copies) so callers can stream them.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path.home() / "data" / "ann"


@dataclass(frozen=True)
class RealDataset:
    name: str
    base: np.ndarray  # (N, D), possibly a float16 memmap
    queries: np.ndarray  # (Q, D) float32


def _queries(path: Path, dim: int, n_queries: int) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, dim)[:n_queries]


def _unit(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)


DBPEDIA_DIM = 1536


def load_dbpedia_queries(n_queries: int) -> np.ndarray:
    """Queries only (no base file needed): for runs on the cached 100k subset."""
    return _queries(DATA_DIR / "dbp_q.f32", DBPEDIA_DIM, n_queries)


def load_dbpedia(n_queries: int) -> RealDataset:
    base = np.load(DATA_DIR / "dbp_base.npy", mmap_mode="r")
    return RealDataset("dbpedia-1536", base, _queries(DATA_DIR / "dbp_q.f32", base.shape[1], n_queries))


def load_glove(n_queries: int, n_base: int | None = None, seed: int = 0) -> RealDataset:
    """`n_base` takes a random (not frequency-ordered) subset for fast iteration."""
    base = np.load(DATA_DIR / "glove_base.npy", mmap_mode="r")
    if n_base is not None:
        rows = np.sort(np.random.default_rng(seed).choice(len(base), size=n_base, replace=False))
        base = base[rows]
    queries = _queries(DATA_DIR / "glove_q.f32", base.shape[1], n_queries)
    name = f"glove-100 ({len(base) // 1000}k)"
    return RealDataset(name, _unit(np.asarray(base)), _unit(queries))
