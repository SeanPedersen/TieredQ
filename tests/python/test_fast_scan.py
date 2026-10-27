"""Full-mode optional fast search preserves configured scoring, array ownership and cache lifecycle."""
from concurrent.futures import ThreadPoolExecutor
import gc

import numpy as np
import pytest

from tieredq import TieredQIndex

DIMENSIONS = 32


@pytest.fixture
def data():
    rows = np.random.default_rng(7).normal(size=(257, DIMENSIONS)).astype(np.float32)
    index = TieredQIndex(dimensions=DIMENSIONS, code_bits=64, mode="full")
    index.add(rows)
    return index, rows


def assert_same(actual, expected):
    for actual_array, expected_array in zip(actual, expected):
        np.testing.assert_array_equal(actual_array, expected_array)


@pytest.mark.parametrize("matrix", [False, True])
def test_fast_is_optional_and_full_search_stays_configured(data, matrix):
    index, rows = data
    queries = rows[:9] if matrix else rows[0]
    expected = index.search(queries, k=10)
    assert not index.has_fast_scan
    assert_same(index.search_fast(queries, k=10), expected)

    index.build_fast_scan()
    assert isinstance(index.has_fast_scan, bool)
    assert_same(index.search(queries, k=10), expected)
    scores, ids = index.search_fast(queries, k=10)
    assert scores.dtype == np.float32 and ids.dtype == np.uint32
    assert scores.shape == ids.shape == ((9, 10) if matrix else (10,))
    assert np.isfinite(scores).all()
    assert np.all(scores[..., :-1] >= scores[..., 1:])
    for actual, reference in zip(ids.reshape(-1, 10), expected[1].reshape(-1, 10)):
        assert len(set(actual) & set(reference)) >= 8
    if not index.has_fast_scan:
        assert_same((scores, ids), expected)


@pytest.mark.parametrize("k", [0, 1, 10, 300])
def test_fast_batch_matches_single_and_preserves_output_rank(data, k):
    index, rows = data
    index.build_fast_scan()
    scores, ids = index.search_fast(rows[:9], k=k)
    for row, query in enumerate(rows[:9]):
        assert_same((scores[row], ids[row]), index.search_fast(query, k=k))
    assert_same(index.search_fast(rows[:1], k=k), tuple(x[None] for x in index.search_fast(rows[0], k=k)))


def test_fast_cache_invalidates_on_append_but_not_empty_or_failed_add(data):
    index, rows = data
    index.build_fast_scan()
    available = index.has_fast_scan
    index.add(rows[:0])
    assert index.has_fast_scan == available
    with pytest.raises(RuntimeError):
        index.add(np.full_like(rows[:1], np.nan))
    assert index.has_fast_scan == available

    index.add(rows[:9])
    assert not index.has_fast_scan
    assert_same(index.search_fast(rows[:3]), index.search(rows[:3]))
    index.build_fast_scan()
    assert index.has_fast_scan == available
    scores, ids = index.search_fast(rows[0], k=len(index))
    assert len(scores) == len(ids) == len(index)
    np.testing.assert_array_equal(np.sort(ids), np.arange(len(index), dtype=np.uint32))


def test_fast_cache_is_derived_after_load_and_outputs_own_memory(data, tmp_path):
    index, rows = data
    index.build_fast_scan()
    path = str(tmp_path / "index.tq")
    index.save(path)
    loaded = TieredQIndex.load(path)
    assert not loaded.has_fast_scan
    assert_same(loaded.search(rows[:3]), index.search(rows[:3]))
    assert_same(loaded.search_fast(rows[:3]), loaded.search(rows[:3]))
    loaded.build_fast_scan()
    actual = loaded.search_fast(rows[:3])
    assert_same(actual, index.search_fast(rows[:3]))
    del index, loaded
    gc.collect()
    assert np.isfinite(actual[0]).all()
    assert np.all(actual[1] < len(rows))


def test_fast_validation_and_worker_exceptions(data):
    index, rows = data
    index.build_fast_scan()
    with pytest.raises(TypeError):
        index.search_fast(rows.astype(np.float64))
    with pytest.raises(ValueError):
        index.search_fast(rows[:, :3].copy())
    with pytest.raises(ValueError):
        index.search_fast(rows.reshape(1, len(rows), DIMENSIONS))
    invalid = rows[:9].copy()
    invalid[7, 0] = np.nan
    with pytest.raises(RuntimeError, match="nonfinite"):
        index.search_fast(invalid)
    assert index.search_fast(rows[:0])[0].shape == (0, 10)
    assert np.isfinite(index.search_fast(rows[0])[0]).all()


def test_fast_cache_requires_training():
    index = TieredQIndex(dimensions=DIMENSIONS, code_bits=64, mode="full")
    assert not index.has_fast_scan
    with pytest.raises(RuntimeError):
        index.build_fast_scan()
    with pytest.raises(RuntimeError):
        index.search_fast(np.zeros(DIMENSIONS, dtype=np.float32))


def test_fast_search_and_cache_build_are_synchronized_with_append(data):
    index, rows = data
    index.build_fast_scan()

    def operation(task):
        if task % 3 == 0:
            index.add(rows[:1])
        elif task % 3 == 1:
            index.build_fast_scan()
        else:
            scores, ids = index.search_fast(rows[:3])
            assert scores.shape == ids.shape == (3, 10)
            assert np.isfinite(scores).all()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(operation, range(24)))
    assert len(index) == len(rows) + 8
