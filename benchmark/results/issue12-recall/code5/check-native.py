"""Distinguish SIFT cutoff ties from score differences in the pruning diagnostic."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path('build-native').resolve()))
from prune_study import (
    build,
    document_terms,
    finish_scores,
    query_terms,
    score_ranks,
    unpack_codes,
)

args = SimpleNamespace(data=Path('data'), queries=200, threads=8, seed=0)
model, packed, probes, native_ids = build(args, 'sift', 4)
codes = unpack_codes(model, packed)
decoded = np.empty(codes.shape, dtype=np.float32, order='F')
for tier in model['tiers']:
    ranks = slice(tier['start'], tier['start'] + tier['dimensions'])
    decoded[:, ranks] = model['levels'][tier['bits']][codes[:, ranks]]
weights, centers, corrections = query_terms(model, probes)
centroid, factor, correction = document_terms(model, packed)
misses = ties = 0
for row in range(len(probes)):
    base = centers[row, centroid] + corrections[row, centroid] * correction
    full = finish_scores(base, factor, score_ranks(decoded, weights[row], range(decoded.shape[1])))
    cutoff = np.partition(full, -10)[-10]
    misses += int(np.count_nonzero(full[native_ids[row]] < cutoff))
    tied = np.flatnonzero(full >= cutoff)
    ordered = tied[np.lexsort((tied, -full[tied]))[:10]]
    ties += len(np.intersect1d(ordered, native_ids[row]))
print(json.dumps({'queries': len(probes), 'native_hits_below_reference_cutoff': misses,
                  'native_overlap_with_score_id_tie_order': ties / (10 * len(probes))}), flush=True)
