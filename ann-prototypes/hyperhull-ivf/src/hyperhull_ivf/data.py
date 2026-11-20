"""Synthetic embedding-like data: anisotropic low-rank "topics" on the unit sphere.

Each topic is a Gaussian around a random center, stretched along a few private
directions (like semantic sub-axes) plus isotropic noise, then L2-normalized.
Two query sets are produced:
  * in-distribution: fresh samples from the same generator (held out)
  * boundary: midpoints between points of two *neighbouring* topics, which is
    where centroid routing is expected to fail most often.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DataConfig:
    n_points: int = 10_560
    dim: int = 64
    n_topics: int = 24
    topic_rank: int = 6
    topic_spread: float = 0.45
    noise: float = 0.06
    n_queries: int = 300
    seed: int = 7


@dataclass(frozen=True)
class Dataset:
    base: np.ndarray  # (N, D) float32, unit norm
    queries_in_dist: np.ndarray  # (Q, D)
    queries_boundary: np.ndarray  # (Q, D)


def _normalize(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _make_topics(cfg: DataConfig, rng: np.random.Generator):
    centers = _normalize(rng.standard_normal((cfg.n_topics, cfg.dim)))
    bases = np.linalg.qr(rng.standard_normal((cfg.n_topics, cfg.dim, cfg.topic_rank)))[0]
    decay = cfg.topic_spread / np.sqrt(1.0 + np.arange(cfg.topic_rank))
    weights = rng.dirichlet(np.full(cfg.n_topics, 2.0))  # uneven topic sizes
    return centers, bases, decay, weights


def _sample(n, topics, cfg: DataConfig, rng: np.random.Generator):
    centers, bases, decay, weights = topics
    labels = rng.choice(cfg.n_topics, size=n, p=weights)
    latent = rng.standard_normal((n, cfg.topic_rank)) * decay
    stretch = np.einsum("ndr,nr->nd", bases[labels], latent)
    noise = rng.standard_normal((n, cfg.dim)) * cfg.noise
    return _normalize(centers[labels] + stretch + noise), labels


def _boundary_queries(points, labels, centers, n, rng: np.random.Generator):
    topic_dist = ((centers[:, None] - centers[None]) ** 2).sum(-1)
    np.fill_diagonal(topic_dist, np.inf)
    nearest_topics = np.argsort(topic_dist, axis=1)[:, :3]
    members = [np.flatnonzero(labels == t) for t in range(len(centers))]

    queries = np.empty((n, points.shape[1]), dtype=np.float32)
    for i in range(n):
        topic_a = rng.integers(len(centers))
        topic_b = rng.choice(nearest_topics[topic_a])
        if len(members[topic_a]) == 0 or len(members[topic_b]) == 0:
            topic_b = topic_a
        a = points[rng.choice(members[topic_a])]
        b = points[rng.choice(members[topic_b])]
        mix = rng.uniform(0.35, 0.65)
        queries[i] = mix * a + (1 - mix) * b
    return _normalize(queries)


def make_dataset(cfg: DataConfig) -> Dataset:
    rng = np.random.default_rng(cfg.seed)
    topics = _make_topics(cfg, rng)
    base, labels = _sample(cfg.n_points, topics, cfg, rng)
    queries_in, _ = _sample(cfg.n_queries, topics, cfg, rng)
    queries_boundary = _boundary_queries(base, labels, topics[0], cfg.n_queries, rng)
    return Dataset(base=base, queries_in_dist=queries_in, queries_boundary=queries_boundary)
