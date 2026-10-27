"""Resume the primary full-accurate SAQ matrix after the Gemini rows completed.

The first run crashed while encoding GloVe at the 1-bit request: SAQ's
``encode_and_fac`` only populated ``base_code.code`` for >1-bit segments, so a
1-bit segment packed an empty code vector.  ``patch_upstream.py`` now moves the
code assignment outside that guard.  This script appends the remaining datasets
without rebuilding the already accepted Gemini rows.
"""
import csv

import numpy as np
from threadpoolctl import threadpool_limits

from adapter import SAQ
from evaluation import evaluate, predictions, recall
from run import HERE, load, pca, save, write, sha

DATASETS = ('glove', 'openai', 'sift')
BITS = (1, 2, 4)
SEEDS = (0, 1, 2)


def main():
    rows = list(csv.DictReader((HERE / 'results.csv').open()))
    completed = {row['dataset'] for row in rows}
    for dataset in DATASETS:
        if dataset in completed:
            print('SKIP', dataset, flush=True)
            continue
        base, queries, truth = load(dataset)
        print('INPUT VERIFIED', dataset, base.shape, flush=True)
        basis, mean, pca_s = pca(base)
        np.savez(HERE / f'pca-{dataset}.npz', basis=basis, mean=mean)
        save(HERE / f'pca-{dataset}.json', dict(
            seconds=pca_s,
            orthogonality_max_error=float(np.max(np.abs(basis.T @ basis - np.eye(base.shape[1]))))))
        print('PCA', dataset, pca_s, flush=True)
        padded = (base.shape[1] + 63) // 64 * 64
        for bits in BITS:
            directory = HERE / f'{dataset}-saq{bits}'
            directory.mkdir(exist_ok=True)
            native_bits = bits * base.shape[1] / padded
            for seed in SEEDS:
                model = directory / f'model-seed{seed}.bin'
                index = SAQ(base, basis, mean, native_bits, seed, model)
                save(directory / f'storage-seed{seed}.json', dict(
                    resident_bytes=index.resident_bytes, disk_bytes=index.disk_bytes,
                    bytes_per_vector=index.bytes_per_vector, model_sha256=sha(model)))
                common = dict(
                    dataset=dataset, d=base.shape[1], base_n=len(base), queries=len(queries),
                    metric='l2' if dataset == 'sift' else 'normalized-cosine-via-l2',
                    codec='saq', config='single-cluster-full-accurate-scan',
                    target_bits=bits, native_padded_bits=native_bits, seed=seed,
                    effective_bytes_per_vector=index.bytes_per_vector,
                    effective_bits_per_dim=index.bytes_per_vector * 8 / base.shape[1],
                    build_s=pca_s + index.build_s, pca_s=pca_s,
                    encode_vps=len(base) / index.encode_seconds)
                rows += evaluate(index, queries, truth, common, directory)
                write(rows)
                index.fullscan = False
                native = predictions(index, queries)
                np.save(directory / f'ids-native-pruned-seed{seed}.npy', native)
                full = np.load(directory / f'ids-seed{seed}.npy')
                save(directory / f'pruning-control-seed{seed}.json', dict(
                    full_recall=recall(truth, full), native_recall=recall(truth, native),
                    identical_rows=int(np.all(full == native, axis=1).sum()), queries=len(queries)))
                index.close()
        print('DATASET COMPLETE', dataset, flush=True)
    meta = dict(
        status='complete',
        resumed_datasets=list(DATASETS),
        binary_sha256=sha(HERE / 'build/libsaq_bench.so'),
        rows=len(rows),
        fix='encode_and_fac now assigns base_code.code for 1-bit segments (upstream crashed in release builds on OOB reads)')
    save(HERE / 'meta-resume.json', meta)
    print('COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
