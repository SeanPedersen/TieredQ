"""Recall@K vs. cost evaluation for fixed-nprobe and adaptive probing.

Exact query→vector distances are computed in query chunks; only the per
(query, list) top-K candidates are kept, so any probe order can be evaluated
by merging cached candidates instead of rescanning. Cost is counted, not timed:
  scanned = vectors whose distance would be computed during list scans
  total   = scanned + routing cost (distance-equivalents)
"""

from dataclasses import dataclass

import numpy as np

from .ivf import IVFIndex, sq_dists
from .routers import Router

K = 10
QUERY_CHUNK = 32


@dataclass(frozen=True)
class CurvePoint:
    param: float  # nprobe, adaptive slack or eps
    recall: float
    scanned: float
    total: float


@dataclass(frozen=True)
class QueryCache:
    gt: np.ndarray  # (Q, K) row ids into index.vectors
    list_topk_d: np.ndarray  # (Q, L, K)
    list_topk_i: np.ndarray  # (Q, L, K)


def _chunk_cache(index: IVFIndex, queries: np.ndarray):
    dists = sq_dists(queries, index.vectors)
    gt = np.argpartition(dists, K, axis=1)[:, :K]
    top_d = np.full((len(queries), index.n_lists, K), np.inf, dtype=np.float32)
    top_i = np.full((len(queries), index.n_lists, K), -1, dtype=np.int64)
    for l in range(index.n_lists):
        sl = index.list_slice(l)
        block = dists[:, sl]
        take = min(K, block.shape[1])
        if take == 0:
            continue
        local = np.argpartition(block, take - 1, axis=1)[:, :take] if block.shape[1] > take else np.argsort(block, 1)
        top_d[:, l, :take] = np.take_along_axis(block, local, 1)
        top_i[:, l, :take] = local + sl.start
    return gt, top_d, top_i


def build_cache(index: IVFIndex, queries: np.ndarray) -> QueryCache:
    parts = [_chunk_cache(index, queries[s : s + QUERY_CHUNK]) for s in range(0, len(queries), QUERY_CHUNK)]
    return QueryCache(*(np.concatenate(p) for p in zip(*parts)))


def score_chunked(router: Router, queries: np.ndarray) -> np.ndarray:
    return np.concatenate([router.score(queries[s : s + QUERY_CHUNK]) for s in range(0, len(queries), QUERY_CHUNK)])


def recall(found: np.ndarray, gt: np.ndarray) -> float:
    return float((found[:, :, None] == gt[:, None, :]).any(-1).sum(1).mean() / K)


def _best_k(cand_d: np.ndarray, cand_i: np.ndarray):
    pick = np.argsort(cand_d, axis=1)[:, :K]
    return np.take_along_axis(cand_d, pick, 1), np.take_along_axis(cand_i, pick, 1)


def fixed_nprobe_curve(index: IVFIndex, cache: QueryCache, router: Router, queries, nprobes) -> list[CurvePoint]:
    order = np.argsort(score_chunked(router, queries), axis=1)
    rows = np.arange(len(queries))[:, None]
    curve = []
    for nprobe in nprobes:
        lists = order[:, :nprobe]
        cand_d = cache.list_topk_d[rows, lists].reshape(len(queries), -1)
        cand_i = cache.list_topk_i[rows, lists].reshape(len(queries), -1)
        _, found = _best_k(cand_d, cand_i)
        scanned = index.sizes[lists].sum(1).mean()
        curve.append(CurvePoint(nprobe, recall(found, cache.gt), scanned, scanned + router.cost_per_query))
    return curve


def _adaptive_run(index: IVFIndex, cache: QueryCache, scores: np.ndarray, slack: float, max_nprobe: int):
    """Probe in score order; stop once the next list's score exceeds
    `slack` × the current K-th best distance."""
    n_q = len(scores)
    order = np.argsort(scores, axis=1)[:, :max_nprobe]
    sorted_scores = np.take_along_axis(scores, order, 1)
    rows = np.arange(n_q)
    best_d = np.full((n_q, K), np.inf, dtype=np.float32)
    best_i = np.full((n_q, K), -1, dtype=np.int64)
    scanned = np.zeros(n_q)
    active = np.ones(n_q, dtype=bool)
    for step in range(order.shape[1]):
        if step > 0:
            active &= sorted_scores[:, step] <= slack * best_d[:, -1]
        if not active.any():
            break
        q = rows[active]
        lists = order[q, step]
        merged_d = np.concatenate([best_d[q], cache.list_topk_d[q, lists]], 1)
        merged_i = np.concatenate([best_i[q], cache.list_topk_i[q, lists]], 1)
        best_d[q], best_i[q] = _best_k(merged_d, merged_i)
        scanned[q] += index.sizes[lists]
    return recall(best_i, cache.gt), scanned.mean()


def adaptive_curve(index, cache, router: Router, queries, slacks, max_nprobe: int) -> list[CurvePoint]:
    scores = score_chunked(router, queries)
    curve = []
    for slack in slacks:
        rec, scanned = _adaptive_run(index, cache, scores, slack, max_nprobe)
        curve.append(CurvePoint(slack, rec, scanned, scanned + router.cost_per_query))
    return curve


def cost_at_recall(curve: list[CurvePoint], target: float, field: str) -> float | None:
    """Cost to reach `target` recall, linearly interpolated along the
    cheapest-first recall frontier (fair to coarse parameter grids)."""
    points = sorted(curve, key=lambda p: p.total)
    frontier, best = [], -1.0
    for p in points:
        if p.recall > best:
            frontier.append(p)
            best = p.recall
    for prev, cur in zip([None, *frontier], frontier):
        if cur.recall < target:
            continue
        if prev is None or cur.recall == prev.recall:
            return getattr(cur, field)
        frac = (target - prev.recall) / (cur.recall - prev.recall)
        return getattr(prev, field) + frac * (getattr(cur, field) - getattr(prev, field))
    return None
