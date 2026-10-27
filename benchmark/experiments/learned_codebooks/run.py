"""Controlled codebook ablation using serialized models and unchanged native kernels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from l2_transform import L2Transform
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native"))
from model import parse_model

K = 10
FIT_LIMIT = 1_000_000
LLOYD_STEPS = 100
FIT_TOLERANCE = 1e-6
BATCH = 32


def replace_levels(model, levels):
    raw = model['prefix'] + struct.pack('<I', len(levels))
    for values in levels:
        raw += struct.pack('<Q', len(values)) + np.asarray(values, '<f4').tobytes()
    return list(raw)


def projected_values(sample, model):
    centers = model['centers']
    assignment = np.empty(len(sample), dtype=np.int64)
    center_norm = np.sum(centers.astype(np.float64)**2, axis=1)
    for start in range(0, len(sample), 512):
        rows = sample[start:start + 512].astype(np.float64)
        assignment[start:start + len(rows)] = np.argmin(center_norm - 2 * rows @ centers.astype(np.float64).T, axis=1)
    residual = sample.astype(np.float64) - centers[assignment]
    return (residual @ model['basis'] - model['mean']) / model['sigma']


def fit_levels(values, weights, initial):
    order = np.argsort(values)
    x, w = values[order], weights[order]
    mass = np.r_[0., np.cumsum(w)]
    sums = np.r_[0., np.cumsum(x * w)]
    levels = initial.astype(np.float64).copy()
    for iteration in range(LLOYD_STEPS):
        boundaries = np.r_[0, np.searchsorted(x, (levels[:-1] + levels[1:]) / 2), len(x)]
        cell_mass = np.diff(mass[boundaries])
        updated = np.divide(np.diff(sums[boundaries]), cell_mass, out=levels.copy(), where=cell_mass > 0)
        delta = np.max(np.abs(updated - levels))
        levels = updated
        if delta < FIT_TOLERANCE:
            break
    if not np.all(np.diff(levels) >= 0) or not np.isfinite(levels).all():
        raise ValueError('invalid learned levels')
    return levels.astype(np.float32), iteration + 1


def evaluate(native, raw, base, probes, truth, heldout, fast):
    codec = native.Codec.load(raw)
    if codec.roundtrip() != raw:
        raise ValueError('model roundtrip mismatch')
    reconstructed = np.empty_like(heldout)
    codec.reconstruct(heldout, reconstructed)
    mse = float(np.mean((heldout.astype(np.float64) - reconstructed)**2))
    index = native.Index(codec, native.Mode.fast if fast else native.Mode.full)
    start = time.perf_counter()
    index.add(base)
    add_s = time.perf_counter() - start
    scores = np.empty((len(probes), K), np.float32)
    ids = np.empty((len(probes), K), np.uint32)

    def search():
        for first in range(0, len(probes), BATCH):
            index.search(probes[first:first+BATCH], scores[first:first+BATCH], ids[first:first+BATCH])
    search()
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        search()
        timings.append(time.perf_counter() - start)
    recall = float(np.mean([len(set(a) & set(b)) / K for a, b in zip(ids, truth)]))
    exact_scores = np.einsum('qd,qkd->qk', probes.astype(np.float64), base[ids].astype(np.float64))
    return dict(recall10=recall, reconstruction_mse=mse, add_s=add_s,
                search_s=timings, qps=len(probes)/float(np.median(timings)),
                returned_score_mse=float(np.mean((scores-exact_scores)**2)),
                vector_storage_bytes=index.storage(), model_memory_bytes=index.model_memory())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', default=['sift', 'glove', 'openai'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--bits', nargs='+', type=int, default=[2, 4])
    parser.add_argument('--queries', type=int, default=200)
    parser.add_argument('--train-rows', type=int, default=8192)
    parser.add_argument('--fit-space', choices=['standardized', 'row-normalized'], default='standardized')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.native.resolve()))
    import codebook_native as native
    native.threads(8)
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    host=os.uname().nodename, threads=8, batch=BATCH, fit_limit=FIT_LIMIT,
                    fitting='sigma-squared weighted standardized tier values; Gaussian initialization',
                    source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*') if p.is_file()})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    with threadpool_limits(limits=8, user_api='blas'):
        for dataset in args.datasets:
            count = 1_000_000 if dataset == 'sift' else 100_000
            with np.load(args.data / f'{dataset}-{count}-1000-v2.npz') as cache:
                original = cache['base']
                queries = cache['queries'][:args.queries]
                truth = cache['truth'][:args.queries]
            transform = L2Transform.fit(original) if dataset == 'sift' else None
            base = transform.base_vectors(original) if transform else original
            probes = transform.queries(queries) if transform else queries
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                selected = rng.choice(len(base), args.train_rows + 2048, replace=False)
                sample = np.ascontiguousarray(base[selected[:args.train_rows]])
                heldout = np.ascontiguousarray(base[selected[args.train_rows:]] )
                for bits in args.bits:
                    for rotate in [True, False]:
                        stem = f'{dataset}-s{seed}-b{bits}-r{int(rotate)}'
                        print(f'TRAIN {stem}', flush=True)
                        start = time.perf_counter()
                        raw = native.train(sample, bits, original.shape[1], rotate, seed)
                        train_s = time.perf_counter()-start
                        model = parse_model(raw)
                        start = time.perf_counter()
                        projected = projected_values(sample, model)
                        row_weight = np.ones(len(sample))
                        if args.fit_space == 'row-normalized':
                            rms = np.sqrt(np.mean(projected**2, axis=1)) + 1e-30
                            projected /= rms[:, None]
                            row_weight = rms**2
                        levels = [v.copy() for v in model['levels']]
                        fitting = []
                        for first, width, tier_bits, _, _ in model['tiers']:
                            values = projected[:, first:first+width].ravel()
                            weights = (row_weight[:, None] * model['sigma'][first:first+width]**2).ravel()
                            if len(values) > FIT_LIMIT:
                                pick = np.random.default_rng(seed).choice(len(values), FIT_LIMIT, replace=False)
                                values, weights = values[pick], weights[pick]
                            levels[tier_bits], iterations = fit_levels(values, weights, levels[tier_bits])
                            fitting.append(dict(bits=int(tier_bits), values=len(values), iterations=iterations))
                        fit_s = time.perf_counter()-start
                        learned = replace_levels(model, levels)
                        if bytes(learned[:len(model['prefix'])]) != model['prefix'] or len(learned)!=len(raw):
                            raise ValueError('fixed-model invariant failed')
                        for variant, recipe in [('gaussian', raw), ('learned', learned)]:
                            (args.output / f'{stem}-{variant}.bin').write_bytes(bytes(recipe))
                            for fast in [False, True]:
                                result = evaluate(native, recipe, base, probes, truth, heldout, fast)
                                row = dict(dataset=dataset, seed=seed, bits_per_original_dimension=bits,
                                           rotate=rotate, variant=variant, fast=fast, fit_space=args.fit_space, base_rows=len(base),
                                           queries=len(probes), train_s=train_s, fit_s=fit_s if variant=='learned' else 0,
                                           tiers=model['tiers'].tolist(), fitting=fitting if variant=='learned' else [], **result)
                                with (args.output / 'results.jsonl').open('a') as out:
                                    out.write(json.dumps(row)+'\n')
                                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
