"""IVF and nested IVF bindings: training from one seeded sample, parity with the flat engine and across tier sources,
the two-step external-tier search, persistence and input validation."""
import numpy as np
import pytest

from tieredq import TieredQIndex, TieredQIVFIndex, TieredQNestedIVFIndex, TieredQPruning

DIMENSIONS = 32
CODE_BITS = 64
PREFIX_BITS = 32
TOTAL_BITS = 4 * DIMENSIONS
ROWS = 1200
QUERIES = 16
CLUSTERS = 8
LISTS = 16
NPROBE = 4
K = 10
CANDIDATES = 40
PADDING_ID = np.iinfo(np.uint32).max
LAYOUTS = ["prefix_and_suffix", "suffix"]


def clustered(rows, seed):
    """Rows around shared cluster centers with decaying per-dimension variance, so lists and PCA both matter."""
    centers = np.random.default_rng(0).normal(scale=3, size=(CLUSTERS, DIMENSIONS))
    decay = np.exp(-4 * np.arange(DIMENSIONS) / DIMENSIONS)
    noise = np.random.default_rng(seed).normal(size=(rows, DIMENSIONS)) * decay
    return (centers[np.arange(rows) % CLUSTERS] + noise).astype(np.float32)


@pytest.fixture(scope="module")
def data():
    return clustered(ROWS, 1), clustered(QUERIES, 2)


def nested(vectors, layout, **options):
    return TieredQNestedIVFIndex.build(vectors, prefix_bits=PREFIX_BITS, total_bits=TOTAL_BITS, lists=LISTS,
                                       tier_rows=layout, **options)


def assert_same(actual, expected):
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)


def recall(ids, vectors, queries):
    truth = np.argsort(-(queries @ vectors.T), axis=1)[:, :K]
    return np.mean([len(set(row) & set(true)) / K for row, true in zip(ids, truth)])


def test_ivf_scanning_every_list_matches_the_flat_index(data):
    vectors, queries = data
    ivf = TieredQIVFIndex(DIMENSIONS, CODE_BITS)
    ivf.build(vectors)
    flat = TieredQIndex(DIMENSIONS, CODE_BITS)
    flat.add(vectors)
    assert len(ivf) == ROWS and ivf.code_bits == flat.code_bits and ivf.metric == "ip"
    scores, ids = ivf.search(queries, k=K, nprobe=ivf.lists)
    flat_scores, flat_ids = flat.search(queries, k=K)
    np.testing.assert_array_equal(ids, flat_ids)
    np.testing.assert_allclose(scores, flat_scores, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("options", [{"exact_fast_scores": True}, {"max_scan_bits": 3}])
def test_ivf_codec_options_match_the_flat_index(data, options):
    """exact_fast_scores and max_scan_bits configure the IVF codec as they configure TieredQIndex."""
    vectors, queries = data
    ivf = TieredQIVFIndex(DIMENSIONS, CODE_BITS, **options)
    ivf.build(vectors)
    flat = TieredQIndex(DIMENSIONS, CODE_BITS, fine_tiers=False, **options)
    flat.add(vectors)
    scores, ids = ivf.search(queries, k=K, nprobe=ivf.lists)
    flat_scores, flat_ids = flat.search(queries, k=K)
    np.testing.assert_array_equal(ids, flat_ids)
    np.testing.assert_allclose(scores, flat_scores, rtol=1e-5, atol=1e-5)


def test_ivf_fine_tiers_match_the_flat_index_and_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    steep = 0.85 ** np.arange(DIMENSIONS)  # steep enough that 8 bits/dim allocates codes past the 8-bit prefix
    vectors = (rng.normal(size=(ROWS, DIMENSIONS)) * steep).astype(np.float32)
    queries = (rng.normal(size=(QUERIES, DIMENSIONS)) * steep).astype(np.float32)
    bits = 8 * DIMENSIONS
    ivf = TieredQIVFIndex(DIMENSIONS, bits, lists=LISTS, fine_tiers=True)
    flat = TieredQIndex(DIMENSIONS, bits)
    ivf.build(vectors)
    flat.add(vectors)
    assert flat.fast_prefix_bits > 0
    scores, ids = ivf.search(queries, k=K, nprobe=ivf.lists)
    flat_scores, flat_ids = flat.search(queries, k=K)
    np.testing.assert_array_equal(ids, flat_ids)
    np.testing.assert_allclose(scores, flat_scores, rtol=1e-5, atol=1e-5)
    path = str(tmp_path / "fine-ivf.tq")
    ivf.save(path)
    assert_same(TieredQIVFIndex.load(path).search(queries, k=K, nprobe=NPROBE), ivf.search(queries, k=K, nprobe=NPROBE))


def test_ivf_search_shapes_padding_pruning_and_persistence(data, tmp_path):
    vectors, queries = data
    ivf = TieredQIVFIndex(DIMENSIONS, CODE_BITS, lists=LISTS)
    ivf.build(vectors)
    assert ivf.lists == LISTS and ivf.storage_bytes > 0
    assignment = ivf.list_assignment()
    assert assignment.dtype == np.uint32 and assignment.shape == (ROWS,) and assignment.max() < LISTS

    scores, ids = ivf.search(queries, k=K, nprobe=NPROBE)
    assert scores.shape == ids.shape == (QUERIES, K) and ids.dtype == np.uint32
    assert np.all(scores[:, :-1] >= scores[:, 1:])
    single = ivf.search(queries[0], k=K, nprobe=NPROBE)
    assert_same(single, (scores[0], ids[0]))
    assert_same(ivf.search(queries, k=K, nprobe=NPROBE, pruning=TieredQPruning()), (scores, ids))

    _, sparse = ivf.search(queries, k=ROWS, nprobe=1)
    assert np.any(sparse == PADDING_ID)

    path = tmp_path / "index.tqivf"
    ivf.save(path)
    loaded = TieredQIVFIndex.load(path)
    assert loaded.lists == LISTS
    assert_same(loaded.search(queries, k=K, nprobe=NPROBE), (scores, ids))
    np.testing.assert_array_equal(loaded.list_assignment(), assignment)


def test_ivf_trained_lists_stay_fixed_across_builds(data):
    vectors, _ = data
    ivf = TieredQIVFIndex(DIMENSIONS, CODE_BITS, lists=LISTS)
    ivf.train_lists(vectors[: ROWS // 2])
    ivf.build(vectors)
    reference = TieredQIVFIndex(DIMENSIONS, CODE_BITS, lists=LISTS)
    reference.train_lists(vectors[: ROWS // 2])
    reference.build(vectors[ROWS // 2:])
    np.testing.assert_array_equal(ivf.list_assignment()[ROWS // 2:], reference.list_assignment())


def test_ivf_rejects_bad_inputs(data):
    vectors, queries = data
    ivf = TieredQIVFIndex(DIMENSIONS, CODE_BITS)
    with pytest.raises(RuntimeError, match="build"):
        ivf.search(queries, k=K, nprobe=NPROBE)
    with pytest.raises(ValueError, match="metric"):
        TieredQIVFIndex(DIMENSIONS, CODE_BITS, metric="hamming")
    with pytest.raises(TypeError):
        ivf.build(vectors.astype(np.float64))
    ivf.build(vectors)
    with pytest.raises(ValueError, match="width"):
        ivf.search(queries[:, :-1].copy(), k=K, nprobe=NPROBE)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_nested_tier_sources_and_round_trips_answer_identically(data, layout, tmp_path):
    vectors, queries = data
    index = nested(vectors, layout)
    assert index.tier_rows == layout and not index.has_tier
    assert index.prefix_bits <= index.total_bits and index.lists == LISTS
    with pytest.raises(RuntimeError, match="tier"):
        index.search(queries, k=K, nprobe=NPROBE, candidates=CANDIDATES)

    tier = index.encode_tier(vectors)
    assert tier.dtype == np.uint8 and tier.shape == (ROWS, index.tier_bytes_per_vector)
    chunked = np.concatenate([index.encode_tier(vectors[first:first + 500], first) for first in range(0, ROWS, 500)])
    np.testing.assert_array_equal(chunked, tier)

    index.attach_tier(tier)
    assert index.has_tier
    expected = index.search(queries, k=K, nprobe=NPROBE, candidates=CANDIDATES)
    assert expected[0].shape == (QUERIES, K)
    assert_same(index.search(queries[0], k=K, nprobe=NPROBE, candidates=CANDIDATES), (expected[0][0], expected[1][0]))

    path = tmp_path / "tier.rows"
    index.write_tier(path, vectors)
    for reads in ("cached", "direct"):
        index.attach_tier(path, reads=reads, queue_depth=8)
        assert_same(index.search(queries, k=K, nprobe=NPROBE, candidates=CANDIDATES), expected)

    restored = TieredQNestedIVFIndex.deserialize(index.serialize())
    restored.attach_tier(str(path))
    assert_same(restored.search(queries, k=K, nprobe=NPROBE, candidates=CANDIDATES), expected)
    saved = tmp_path / "index.tqnested"
    index.save(saved)
    loaded = TieredQNestedIVFIndex.load(saved)
    assert loaded.tier_rows == layout and not loaded.has_tier
    loaded.attach_tier(tier)
    assert_same(loaded.search(queries, k=K, nprobe=NPROBE, candidates=CANDIDATES), expected)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_two_step_search_over_an_external_tier_matches_search(data, layout):
    vectors, queries = data
    index = nested(vectors, layout)
    tier = index.encode_tier(vectors)
    index.attach_tier(tier)
    expected = index.search(queries, k=K, nprobe=NPROBE, candidates=CANDIDATES)

    ids = index.candidates(queries, CANDIDATES, nprobe=NPROBE)
    assert ids.shape == (QUERIES, CANDIDATES)
    fetched = np.zeros(ids.shape + (index.tier_bytes_per_vector,), np.uint8)
    found = ids != PADDING_ID
    fetched[found] = tier[ids[found]]
    assert_same(index.rescore(queries, ids, fetched, k=K), expected)
    assert_same(index.rescore(queries[0], ids[0], fetched[0], k=K), (expected[0][0], expected[1][0]))

    with pytest.raises(ValueError, match="tier row"):
        index.rescore(queries, ids, fetched[:, :, :-1].copy(), k=K)
    out_of_range = ids.copy()
    out_of_range[0, 0] = ROWS
    with pytest.raises(RuntimeError, match="range"):
        index.rescore(queries, out_of_range, fetched, k=K)


def test_nested_rescoring_beats_the_prefix_scan(data):
    vectors, queries = data
    index = nested(vectors, "suffix")
    index.attach_tier(index.encode_tier(vectors))
    prefix_ids = index.candidates(queries, K, nprobe=NPROBE)
    _, nested_ids = index.search(queries, k=K, nprobe=NPROBE, candidates=CANDIDATES)
    assert recall(nested_ids, vectors, queries) > recall(prefix_ids, vectors, queries)


def test_explicit_sample_trains_the_same_model_and_fixes_the_lists(data):
    vectors, queries = data
    implicit = nested(vectors, "suffix")
    explicit = nested(vectors, "suffix", sample=vectors)
    np.testing.assert_array_equal(explicit.encode_tier(vectors), implicit.encode_tier(vectors))

    earlier = vectors[: ROWS // 2]
    later = nested(vectors[ROWS // 2:], "suffix", sample=earlier)
    reference = TieredQIVFIndex(DIMENSIONS, PREFIX_BITS, lists=LISTS)
    reference.train_lists(earlier)
    reference.build(vectors[ROWS // 2:])
    later.attach_tier(later.encode_tier(vectors[ROWS // 2:]))
    assert_same(later.candidates(queries, K, nprobe=NPROBE), reference.search(queries, k=K, nprobe=NPROBE)[1])


def test_nested_rejects_bad_inputs(data, tmp_path):
    vectors, _ = data
    with pytest.raises(ValueError, match="tier_rows"):
        nested(vectors, "prefix")
    index = nested(vectors, "suffix")
    with pytest.raises(RuntimeError, match="tier"):
        index.attach_tier(np.zeros((ROWS, 1), np.uint8))
    with pytest.raises(ValueError, match="reads"):
        index.attach_tier(tmp_path / "tier.rows", reads="mmap")
    with pytest.raises(ValueError, match="one row per indexed vector"):
        index.write_tier(tmp_path / "tier.rows", vectors[:-1])


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("metric", ["ip", "cosine"])
@pytest.mark.parametrize("prefix_bits", [1, 2 * DIMENSIONS])
def test_nested_fine_suffix_preserves_prefix_and_round_trips(tmp_path, layout, metric, prefix_bits):
    rng = np.random.default_rng(3)
    vectors = (rng.normal(size=(ROWS, DIMENSIONS)) * 0.85 ** np.arange(DIMENSIONS)).astype(np.float32)
    queries = (rng.normal(size=(QUERIES, DIMENSIONS)) * 0.85 ** np.arange(DIMENSIONS)).astype(np.float32)
    options = dict(prefix_bits=prefix_bits, total_bits=8 * DIMENSIONS, lists=LISTS,
                   tier_rows=layout, metric=metric)
    baseline = TieredQNestedIVFIndex.build(vectors, **options)
    fine = TieredQNestedIVFIndex.build(vectors, fine_tiers=True, **options)
    assert baseline.max_refinement_bits <= 8 < fine.max_refinement_bits <= 12
    ids = fine.candidates(queries, k=CANDIDATES, nprobe=NPROBE)
    np.testing.assert_array_equal(ids, baseline.candidates(queries, k=CANDIDATES, nprobe=NPROBE))
    tier = fine.encode_tier(vectors)
    fine.attach_tier(tier)
    expected = fine.search(queries, k=K, nprobe=NPROBE, candidates=CANDIDATES)
    assert np.isfinite(expected[0]).all()
    assert_same(fine.rescore(queries, ids, tier[ids], k=K), expected)
    restored = TieredQNestedIVFIndex.deserialize(fine.serialize())
    assert restored.max_refinement_bits == fine.max_refinement_bits
    np.testing.assert_array_equal(restored.encode_tier(vectors), tier)
    restored.attach_tier(tier)
    assert_same(restored.search(queries, k=K, nprobe=NPROBE, candidates=CANDIDATES), expected)
    path = tmp_path / "fine-tier.rows"
    fine.write_tier(path, vectors)
    restored.attach_tier(path)
    assert_same(restored.search(queries, k=K, nprobe=NPROBE, candidates=CANDIDATES), expected)


@pytest.mark.parametrize("cap", [0, 7, 13])
def test_nested_fine_suffix_rejects_invalid_cap(data, cap):
    with pytest.raises(RuntimeError, match="max tier bits"):
        nested(data[0], "suffix", fine_tiers=True, max_tier_bits=cap)
