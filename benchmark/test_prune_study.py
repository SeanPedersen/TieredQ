"""Check pruning diagnostics against sequential FP32 scores and signed document factors."""
import numpy as np
from prune_study import (
    document_terms,
    finish_scores,
    query_terms,
    score_ranks,
    study,
    unpack_codes,
)


def test_prefix_continuation_preserves_fp32_rounding():
    decoded = np.array([[2**24, 1, -(2**24), 1]], dtype=np.float32)
    weights = np.ones(4, dtype=np.float32)
    prefix = score_ranks(decoded, weights, range(2))
    full = score_ranks(decoded, weights, range(4))
    continued = score_ranks(decoded, weights, range(2, 4), prefix)
    np.testing.assert_array_equal(continued.view(np.uint32), full.view(np.uint32))
    assert full[0] == 1


def test_bounds_and_survivor_scores_with_negative_factors():
    rng = np.random.default_rng(12)
    count, dimensions = 513, 4
    model = {
        "dimensions": dimensions, "code_bytes": 1,
        "sigma": np.ones(dimensions, dtype=np.float32),
        "mean": np.zeros(dimensions, dtype=np.float32),
        "basis": np.eye(dimensions, dtype=np.float32),
        "centroids": np.ones((1, dimensions), dtype=np.float32),
        "levels": [np.empty(0), np.array([-1, 1], dtype=np.float32),
                   np.array([-2, -0.5, 0.5, 2], dtype=np.float32)],
        "tiers": [{"start": 0, "dimensions": 2, "bits": 2, "bit_offset": 0},
                  {"start": 2, "dimensions": 2, "bits": 1, "bit_offset": 4}],
    }
    packed = np.zeros((count, 6), dtype=np.uint8)
    packed[:, 0] = rng.integers(0, 64, count, dtype=np.uint8)
    terms = np.column_stack([rng.choice([-1, 1], count), rng.normal(size=count)]).astype("<f2")
    packed[:, 2:] = terms.view(np.uint8)
    codes = unpack_codes(model, packed)
    decoded = np.column_stack([model["levels"][2][codes[:, :2]], model["levels"][1][codes[:, 2:]]])
    probes = rng.normal(size=(5, dimensions)).astype(np.float32)
    weights, centers, corrections = query_terms(model, probes)
    centroid, factor, correction = document_terms(model, packed)
    native_ids = []
    for row in range(len(probes)):
        base = centers[row, centroid] + corrections[row, centroid] * correction
        scores = finish_scores(base, factor, score_ranks(decoded, weights[row], range(dimensions)))
        native_ids.append(np.argpartition(scores, -10)[-10:])
    prefixes, overlap, checks = study(model, packed, probes, np.asarray(native_ids))
    assert overlap == 1
    assert checks > 0
    assert len(prefixes) == 1
