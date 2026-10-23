"""Exercise the index surface: implicit training, ownership, persistence and concurrency."""
from concurrent.futures import ThreadPoolExecutor
import gc

import numpy as np
import pytest

from tieredq import TieredQIndex

DIMENSIONS = 16
CODE_BITS = 32


def vectors(rows=128, seed=42):
    return np.random.default_rng(seed).normal(size=(rows, DIMENSIONS)).astype(np.float32)


def built(rows):
    index = TieredQIndex(dimensions=DIMENSIONS, code_bits=CODE_BITS)
    index.add(rows)
    return index


def test_add_trains_on_first_batch_and_search_scores_before_ids():
    rows = vectors()
    index = TieredQIndex(dimensions=DIMENSIONS, code_bits=CODE_BITS)
    index.add(rows[:17])
    index.add(rows[17:])
    assert len(index) == len(rows)
    assert index.dimensions == DIMENSIONS and index.code_bits == CODE_BITS
    assert index.bytes_per_vector > 0

    scores, ids = index.search(rows[:2], k=len(rows) + 1)
    assert scores.dtype == np.float32 and scores.shape == (2, len(rows))
    assert ids.dtype == np.uint32 and ids.shape == scores.shape
    assert np.all(scores[:, :-1] >= scores[:, 1:])
    assert ids[0, 0] == 0 and ids[1, 0] == 1


def test_single_vector_query_keeps_rank():
    rows = vectors()
    index = built(rows)
    scores, ids = index.search(rows[0], k=10)
    assert scores.shape == ids.shape == (10,)
    matrix_scores, matrix_ids = index.search(rows[:1], k=10)
    np.testing.assert_array_equal(ids, matrix_ids[0])
    np.testing.assert_array_equal(scores, matrix_scores[0])


def test_persistence_and_owned_outputs(tmp_path):
    rows = vectors()
    index = built(rows)
    scores, ids = index.search(rows[:1], k=len(rows))
    path = str(tmp_path / "index.tq")
    index.save(path)

    loaded = TieredQIndex.load(path)
    assert len(loaded) == len(rows)
    assert loaded.dimensions == DIMENSIONS and loaded.code_bits == CODE_BITS
    actual_scores, actual_ids = loaded.search(rows[:1], k=len(rows))
    np.testing.assert_array_equal(actual_ids, ids)
    np.testing.assert_array_equal(actual_scores, scores)

    del loaded, index
    gc.collect()
    assert np.isfinite(scores).all() and ids[0, 0] == 0


def test_first_non_empty_add_trains():
    rows = vectors()
    index = TieredQIndex(dimensions=DIMENSIONS, code_bits=CODE_BITS)
    assert len(index) == 0
    with pytest.raises(RuntimeError):
        index.search(rows[:1])
    with pytest.raises(RuntimeError):
        index.save("unused.tq")

    index.add(rows[:0])
    with pytest.raises(RuntimeError):
        index.search(rows[:1])
    with pytest.raises(RuntimeError):
        index.add(rows[:1])
    with pytest.raises(RuntimeError):
        index.search(rows[:1])

    index.add(rows)
    assert len(index) == len(rows)


def test_default_centroid_count_fits_small_samples():
    rows = vectors(rows=32)
    index = TieredQIndex(dimensions=DIMENSIONS, code_bits=CODE_BITS)
    index.add(rows)
    assert len(index) == len(rows)


def test_validation_and_empty_arrays():
    rows = vectors()
    index = built(rows)

    with pytest.raises(TypeError):
        index.add(rows.astype(np.float64))
    with pytest.raises(TypeError):
        index.add(rows[:, ::2])
    with pytest.raises(TypeError):
        index.search(rows[:1].astype(np.float64))
    with pytest.raises(ValueError):
        index.add(rows[:, :3].copy())
    with pytest.raises(ValueError):
        index.search(np.zeros((1, DIMENSIONS, 1), dtype=np.float32))
    with pytest.raises(RuntimeError):
        TieredQIndex(dimensions=DIMENSIONS, code_bits=0)
    with pytest.raises(RuntimeError):
        TieredQIndex(dimensions=DIMENSIONS, code_bits=DIMENSIONS * 8 + 1)

    index.add(rows[:0])
    assert len(index) == len(rows)
    scores, ids = index.search(rows[:0], k=5)
    assert scores.shape == ids.shape == (0, 5)

    readonly = rows.copy()
    readonly.flags.writeable = False
    index.add(readonly)


def test_nonfinite_first_batch_leaves_index_untrained():
    rows = vectors()
    index = TieredQIndex(dimensions=DIMENSIONS, code_bits=CODE_BITS)
    with pytest.raises(RuntimeError):
        index.add(np.full_like(rows, np.nan))
    assert len(index) == 0
    index.add(rows)
    assert len(index) == len(rows)


def test_concurrent_search_and_append():
    rows = vectors()
    index = built(rows)

    def append_or_search(task):
        if task % 3 == 0:
            index.add(rows[:1])
            return
        scores, ids = index.search(rows[0], k=10)
        assert len(scores) == len(ids) == 10

    tasks = list(range(24))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(append_or_search, tasks))
    assert len(index) == len(rows) + sum(task % 3 == 0 for task in tasks)
