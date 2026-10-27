"""Index mode selection, rejected legacy options, and persisted representation semantics."""
import numpy as np
import pytest
from tieredq import TieredQIndex

DIMENSIONS = 8
CODE_BITS = 16
ROWS = 32


@pytest.mark.parametrize("options,expected", [({}, "fast"), ({"mode": "fast"}, "fast"), ({"mode": "full"}, "full")])
def test_mode_selection_persistence_and_append(options, expected, tmp_path):
    vectors = np.random.default_rng(73).normal(size=(ROWS, DIMENSIONS)).astype(np.float32)
    index = TieredQIndex(DIMENSIONS, CODE_BITS, **options)
    assert index.mode == expected
    assert not hasattr(index, "fast_mode")
    with pytest.raises(AttributeError):
        index.mode = "full"
    path = str(tmp_path / "index.tq")
    index.add(vectors)
    before = index.search(vectors[:2])
    index.save(path)
    loaded = TieredQIndex.load(path)
    assert loaded.mode == expected
    for actual, reference in zip(loaded.search(vectors[:2]), before):
        np.testing.assert_array_equal(actual, reference)
    loaded.add(vectors[:2])
    assert len(loaded) == ROWS + 2
    assert loaded.mode == expected
    assert loaded.has_fast_scan == (expected == "fast")


@pytest.mark.parametrize("mode", ["", "normal", "FAST", "invalid"])
def test_invalid_mode_is_rejected_before_training(mode):
    with pytest.raises(ValueError, match="mode must be 'fast' or 'full'"):
        TieredQIndex(DIMENSIONS, CODE_BITS, mode=mode)


@pytest.mark.parametrize("options", [{"fast_mode": True}, {"fast_mode": False}, {"fast": True}, {"mode": True}, {"mode": None}, {"mode": "full", "fast_mode": False}])
def test_boolean_constructor_api_is_removed(options):
    with pytest.raises(TypeError):
        TieredQIndex(DIMENSIONS, CODE_BITS, **options)
