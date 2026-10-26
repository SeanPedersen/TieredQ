"""Pinned Gemini document-vector dataset with auditable held-out document queries."""
import hashlib
import json
from pathlib import Path
import numpy as np

DIMENSIONS = 768
DOCUMENTS = 100_000
HELD_OUT = 1_000
SEED = 0


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def prepare_gemini(root, max_base=None, query_count=HELD_OUT):
    from data import SOURCES, download, normalize, ground_truth
    import pyarrow.parquet as pq
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    source = SOURCES['gemini']
    arrays = []
    checksums = {}
    for relative, checksum in source['files'].items():
        path = root / Path(relative).name
        url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{relative}"
        download(url, path, checksum)
        table = pq.read_table(path)
        candidates = [name for name in table.column_names if 'embed' in name.lower()]
        if len(candidates) != 1:
            raise ValueError(f'Cannot identify embedding column: {table.schema}')
        rows = table[candidates[0]].to_pylist()
        if any(len(row) != DIMENSIONS for row in rows):
            raise ValueError('Ragged or unexpected embedding dimensions')
        arrays.append(np.asarray(rows, dtype=np.float32))
        checksums[relative] = checksum
    vectors = np.concatenate(arrays)
    if vectors.shape != (DOCUMENTS, DIMENSIONS) or not np.isfinite(vectors).all():
        raise ValueError(f'Unexpected dataset: {vectors.shape}')
    query_ids = np.sort(np.random.default_rng(SEED).choice(len(vectors), HELD_OUT, replace=False))
    corpus_ids = np.setdiff1d(np.arange(len(vectors)), query_ids)
    assert not np.intersect1d(corpus_ids, query_ids).size
    if not 1 <= query_count <= HELD_OUT:
        raise ValueError('Invalid query count')
    selected_corpus = corpus_ids[:max_base] if max_base else corpus_ids
    selected_queries = query_ids[:query_count]
    base, queries = normalize(vectors[selected_corpus]), normalize(vectors[selected_queries])
    print(f'Gemini verified {vectors.shape}; corpus={len(base)} queries={len(queries)}', flush=True)
    truth = ground_truth(base, queries, False)
    np.save(root/'gemini-corpus-row-ids.npy', selected_corpus)
    np.save(root/'gemini-query-row-ids.npy', selected_queries)
    manifest = dict(source=source, source_checksums=checksums, original_shape=list(vectors.shape), finite=True,
                    split_seed=SEED, corpus_n=len(base), query_n=len(queries),
                    query_semantics='held-out document vectors, not independently embedded search queries',
                    normalization='FP32 L2 normalization for both corpus and queries',
                    truth='exhaustive chunked FP32 dot product top-10',
                    array_sha256={key:digest(value) for key,value in [('raw',vectors),('base',base),('queries',queries),('truth',truth),('corpus_ids',selected_corpus),('query_ids',selected_queries)]})
    (root/'gemini-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    np.savez(root/'gemini-prepared.npz',base=base,queries=queries,truth=truth)
    return base, queries, truth
