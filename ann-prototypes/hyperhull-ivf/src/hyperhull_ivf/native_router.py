"""Export a two-level index to the arrays of the C++ HullRouter (hyperhull_native).

Global PCA (pca_dims, fit on a data sample); coarse centroids kept as float in the first coarse_dims;
fine centroids (first fine_dims) and per-vector sketches (all pca_dims) stored as 4-bit per-dim scalar
codes (each with its own lo/step), packed for the AVX2 kernel: within each 64-dim block, byte j = code[j] | code[32+j] << 4.
Per-vector terms: sketch_bias = ||ẑ_x||² + o_x², sketch_resid = o_x (exact residual norm outside the PCA).
"""

import time
from dataclasses import dataclass

import numpy as np

from .two_stage import TwoLevel

CODE_LEVELS = 15
BLOCK_DIMS = 64
PCA_SAMPLE = 50_000
CHUNK = 65_536


@dataclass(frozen=True)
class SQ4:
    codes: np.ndarray  # (n, dims/2) packed uint8
    lo: np.ndarray
    step: np.ndarray
    norm2: np.ndarray  # ||dequantized||²


def pack4(codes: np.ndarray) -> np.ndarray:
    n, dims = codes.shape
    blocks = codes.reshape(n, dims // BLOCK_DIMS, 2, BLOCK_DIMS // 2)
    return (blocks[:, :, 0] | (blocks[:, :, 1] << 4)).reshape(n, dims // 2).astype(np.uint8)


def pack2(codes: np.ndarray) -> np.ndarray:
    """byte j = c[j] | c[Q+j] << 2 | c[2Q+j] << 4 | c[3Q+j] << 6, Q = dims/4."""
    n, dims = codes.shape
    q = codes.reshape(n, 4, dims // 4)
    return (q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)).astype(np.uint8)


TWO_BIT_SPAN = 1.494  # Gaussian-optimal 4-level uniform quantizer: levels at mean ± 0.498σ, ± 1.494σ


def sq2(z: np.ndarray) -> SQ4:
    """Per-dim 2-bit uniform quantizer over mean ± TWO_BIT_SPAN·σ (clips the tails)."""
    mu, sigma = z.mean(0), z.std(0) + 1e-12
    lo = mu - TWO_BIT_SPAN * sigma
    step = 2 * TWO_BIT_SPAN * sigma / 3
    codes = np.clip(np.rint((z - lo) / step), 0, 3).astype(np.uint8)
    deq = lo + codes * step
    return SQ4(pack2(codes), lo.astype(np.float32), step.astype(np.float32), (deq * deq).sum(1).astype(np.float32))


def sq4(z: np.ndarray) -> SQ4:
    lo = z.min(0)
    step = np.maximum(z.max(0) - lo, 1e-12) / CODE_LEVELS
    codes = np.clip(np.rint((z - lo) / step), 0, CODE_LEVELS).astype(np.uint8)
    deq = lo + codes * step
    return SQ4(pack4(codes), lo.astype(np.float32), step.astype(np.float32), (deq * deq).sum(1).astype(np.float32))


def pca_basis(centered: np.ndarray, dims: int) -> np.ndarray:
    """Top `dims` principal directions (dims × D, by decreasing variance) of centered rows.

    Eigenvectors of the D × D scatter matrix: one GEMM + eigh instead of an SVD of the full sample
    (50k × 1536: SVD ~57 s wall on hartmut, dominating the whole build). Same subspace; signs may differ.
    """
    scatter = centered.T @ centered
    _, vectors = np.linalg.eigh(scatter.astype(np.float64))
    return np.ascontiguousarray(vectors[:, ::-1][:, :dims].T, dtype=np.float32)


def native_arrays(index: TwoLevel, pca_dims: int = 128, coarse_dims: int = 64, fine_dims: int = 128,
                  sketch_bits: int = 4, gamma: float = 0.5, seed: int = 0, timings: dict | None = None) -> dict:
    """timings (optional) receives wall seconds of the stages: pca_s, project_s, quantize_s."""
    clock = time.perf_counter()
    vectors = index.vectors
    rng = np.random.default_rng(seed)
    sample = vectors[rng.choice(len(vectors), size=min(PCA_SAMPLE, len(vectors)), replace=False)]
    mean = sample.mean(0).astype(np.float32)
    proj = pca_basis(sample - mean, pca_dims)
    pca_s = time.perf_counter() - clock

    clock = time.perf_counter()
    z = np.empty((len(vectors), pca_dims), dtype=np.float32)
    resid2 = np.empty(len(vectors), dtype=np.float32)
    for s in range(0, len(vectors), CHUNK):
        centered = vectors[s : s + CHUNK] - mean
        z[s : s + CHUNK] = centered @ proj.T
        resid2[s : s + CHUNK] = np.maximum((centered * centered).sum(1) - (z[s : s + CHUNK] ** 2).sum(1), 0)
    project_s = time.perf_counter() - clock

    clock = time.perf_counter()
    sketch = sq4(z) if sketch_bits == 4 else sq2(z)
    fine = sq4((index.fine_centroids - mean) @ proj[:fine_dims].T)
    if timings is not None:
        timings.update(pca_s=pca_s, project_s=project_s, quantize_s=time.perf_counter() - clock)
    return dict(
        mean=mean,
        proj=proj,
        proj_offset=(proj @ mean).astype(np.float32),
        coarse_z=np.ascontiguousarray(((index.coarse_centroids - mean) @ proj.T)[:, :coarse_dims], dtype=np.float32),
        fine_codes=fine.codes, fine_lo=fine.lo, fine_step=fine.step, fine_bias=fine.norm2,
        sketch_codes=sketch.codes, sketch_lo=sketch.lo, sketch_step=sketch.step,
        sketch_bias=(sketch.norm2 + resid2).astype(np.float32),
        sketch_resid=np.sqrt(resid2).astype(np.float32),
        vectors=np.ascontiguousarray(vectors, dtype=np.float32),
        ids=index.ids,
        coarse_offsets=index.coarse_offsets.astype(np.int64),
        fine_offsets=index.fine_offsets.astype(np.int64),
        fine_per_coarse=index.m,
        gamma=gamma,
    )


def extra_bytes(arrays: dict) -> int:
    """Index memory beyond the raw float vectors (codes, centroids, per-vector terms, ids, offsets)."""
    return sum(v.nbytes for k, v in arrays.items() if isinstance(v, np.ndarray) and k != "vectors")


def make_router(arrays: dict, prefix_dims: int = 0, fine_used_dims: int = 0):
    """prefix_dims > 0 enables the two-pass sketch (search(..., refine=R)); 0 = single pass."""
    from hyperhull_native import HullRouter

    return HullRouter(**arrays, prefix_dims=prefix_dims, fine_used_dims=fine_used_dims)


def export(arrays: dict, queries: np.ndarray, gt: np.ndarray, out_dir) -> None:
    """Write router arrays + queries + ground truth as raw files for the standalone C++ bench."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {**{k: v for k, v in arrays.items() if isinstance(v, np.ndarray)},
              "queries": np.ascontiguousarray(queries, dtype=np.float32), "gt": np.ascontiguousarray(gt, dtype=np.int64)}
    lines = [f"fine_per_coarse {arrays['fine_per_coarse']}", f"gamma {arrays['gamma']}"]
    for name, a in tables.items():
        np.ascontiguousarray(a).tofile(out / f"{name}.bin")
        lines.append(f"array {name} {a.dtype} " + " ".join(map(str, a.shape)))
    (out / "meta.txt").write_text("\n".join(lines) + "\n")
