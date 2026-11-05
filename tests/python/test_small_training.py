"""Small and degenerate training sets must remain usable after basis rounding and for rows added later."""
import numpy as np
import pytest

from tieredq import TieredQIndex


@pytest.mark.parametrize("metric", ["ip", "cosine"])
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("count", [200, 400, 512, 1000])
def test_rank_deficient_fp16_training(metric, seed, count):
    dimensions = 512
    rows = np.random.default_rng(seed).standard_normal((count, dimensions)).astype(np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    rows = rows.astype(np.float16).astype(np.float32)
    index = TieredQIndex(
        dimensions=dimensions, code_bits=4 * dimensions, metric=metric, fine_tiers=False
    )
    index.add(rows)
    scores, ids = index.search(rows[:2], k=10)
    assert len(index) == count
    assert np.isfinite(scores).all()
    assert ((ids >= 0) & (ids < count)).all()


def _degenerate_rows(rng, kind, count, dimensions):
    if kind == "zero":
        return np.zeros((count, dimensions), dtype=np.float32)
    if kind == "duplicate":
        return np.repeat(rng.standard_normal((1, dimensions)), count, axis=0).astype(np.float32)
    return (rng.standard_normal((count, 3)) @ rng.standard_normal((3, dimensions))).astype(np.float32)


@pytest.mark.parametrize("kind", ["zero", "duplicate", "low_rank"])
@pytest.mark.parametrize("mode", ["fast", "full"])
def test_degenerate_training_and_persistence(kind, mode, tmp_path):
    dimensions, count = 64, 32
    rng = np.random.default_rng(42)
    rows = _degenerate_rows(rng, kind, count, dimensions)
    index = TieredQIndex(dimensions=dimensions, code_bits=4 * dimensions, mode=mode, fine_tiers=False)
    index.add(rows)
    expected = index.search(rows[:2], k=10)
    assert np.isfinite(expected[0]).all()
    path = str(tmp_path / "small.tq")
    index.save(path)
    loaded = TieredQIndex.load(path)
    for actual, wanted in zip(loaded.search(rows[:2], k=10), expected):
        np.testing.assert_array_equal(actual, wanted)


def _unit_rows(rng, count, dimensions):
    rows = rng.standard_normal((count, dimensions)).astype(np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _recall_at(index, rows, queries, k):
    truth = np.argsort(-(queries @ rows.T), axis=1)[:, :k]
    _, ids = index.search(queries, k=k)
    return np.mean([len(set(found) & set(wanted)) / k for found, wanted in zip(ids, truth)])


@pytest.mark.parametrize("metric", ["ip", "cosine", "l2"])
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("first,then", [(2, 1), (2, 50), (100, 100), (300, 300), (200, 400)])
def test_add_after_rank_deficient_training(metric, seed, first, then):
    """#35: rows added later carry energy in directions the small training sample never varied in."""
    dimensions = 512
    rows = _unit_rows(np.random.default_rng(seed), first + then, dimensions).astype(np.float16).astype(np.float32)
    index = TieredQIndex(dimensions=dimensions, code_bits=4 * dimensions, metric=metric, fine_tiers=False)
    index.add(rows[:first])
    index.add(rows[first:])
    scores, ids = index.search(rows[-2:], k=10)
    assert len(index) == first + then
    assert np.isfinite(scores).all()
    assert ((ids >= 0) & (ids < first + then)).all()


def test_added_rows_stay_searchable_after_rank_deficient_training():
    # Before #35 rows added to this model scored near-randomly (recall@10 about 0.05); now about 0.6.
    minimum_recall = 0.4
    dimensions, first, then, k = 512, 300, 300, 10
    rng = np.random.default_rng(0)
    rows, queries = _unit_rows(rng, first + then, dimensions), _unit_rows(rng, 50, dimensions)
    index = TieredQIndex(dimensions=dimensions, code_bits=4 * dimensions, metric="cosine", fine_tiers=False)
    index.add(rows[:first])
    index.add(rows[first:])
    assert _recall_at(index, rows, queries, k) >= minimum_recall


@pytest.mark.parametrize("kind", ["zero", "duplicate", "low_rank"])
@pytest.mark.parametrize("metric", ["ip", "cosine", "l2"])
def test_add_after_degenerate_training(kind, metric):
    dimensions, count = 64, 32
    rng = np.random.default_rng(42)
    rows = _degenerate_rows(rng, kind, count, dimensions)
    index = TieredQIndex(dimensions=dimensions, code_bits=4 * dimensions, metric=metric, fine_tiers=False)
    index.add(rows)
    added = _unit_rows(rng, 200, dimensions)
    index.add(added)
    scores, _ = index.search(added[:2], k=10)
    assert np.isfinite(scores).all()
