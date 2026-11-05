"""Small and degenerate training sets must remain usable after basis rounding."""
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


@pytest.mark.parametrize("kind", ["zero", "duplicate", "low_rank"])
@pytest.mark.parametrize("mode", ["fast", "full"])
def test_degenerate_training_and_persistence(kind, mode, tmp_path):
    dimensions, count = 64, 32
    rng = np.random.default_rng(42)
    if kind == "zero":
        rows = np.zeros((count, dimensions), dtype=np.float32)
    elif kind == "duplicate":
        rows = np.repeat(rng.standard_normal((1, dimensions)), count, axis=0).astype(np.float32)
    else:
        rows = (rng.standard_normal((count, 3)) @ rng.standard_normal((3, dimensions))).astype(np.float32)
    index = TieredQIndex(dimensions=dimensions, code_bits=4 * dimensions, mode=mode, fine_tiers=False)
    index.add(rows)
    expected = index.search(rows[:2], k=10)
    assert np.isfinite(expected[0]).all()
    path = str(tmp_path / "small.tq")
    index.save(path)
    loaded = TieredQIndex.load(path)
    for actual, wanted in zip(loaded.search(rows[:2], k=10), expected):
        np.testing.assert_array_equal(actual, wanted)
