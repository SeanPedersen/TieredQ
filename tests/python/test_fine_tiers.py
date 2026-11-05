"""Fine-tier configuration of the flat index (#26): option validation, forced fine tiers, persistence and rejection."""
import numpy as np
import pytest
from tieredq import TieredQIndex

DIMENSIONS = 32
BITS_PER_DIM = 8
ROWS = 1024
QUERIES = 8
K = 10
DECAY = 0.85  # per-dimension spectrum decay steep enough that 8 bits/dim allocates codes past 8 bits


def steep_rows(count, seed):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(count, DIMENSIONS)) * DECAY ** np.arange(DIMENSIONS)).astype(np.float32)


@pytest.mark.parametrize("options", [{"fine_tiers": "yes"}, {"fine_tiers": "AUTO"}])
def test_invalid_fine_tier_mode_is_rejected(options):
    with pytest.raises(ValueError, match="fine_tiers must be"):
        TieredQIndex(DIMENSIONS, BITS_PER_DIM * DIMENSIONS, **options)


@pytest.mark.parametrize("options", [{"max_scan_bits": 9}, {"max_tier_bits": 13}, {"max_tier_bits": 0}])
def test_out_of_range_caps_are_rejected(options):
    with pytest.raises(RuntimeError):
        TieredQIndex(DIMENSIONS, BITS_PER_DIM * DIMENSIONS, **options)


def test_forced_fine_tiers_search_and_round_trip(tmp_path):
    vectors, queries = steep_rows(ROWS, 26), steep_rows(QUERIES, 27)
    plain = TieredQIndex(DIMENSIONS, BITS_PER_DIM * DIMENSIONS, fine_tiers="off")
    fine = TieredQIndex(DIMENSIONS, BITS_PER_DIM * DIMENSIONS, fine_tiers="on")
    plain.add(vectors)
    fine.add(vectors)
    assert fine.fast_prefix_bits == BITS_PER_DIM and plain.fast_prefix_bits == 0
    assert fine.bytes_per_vector == plain.bytes_per_vector
    scores, ids = fine.search(queries, K)
    assert ids.shape == (QUERIES, K) and np.all(np.diff(scores, axis=1) <= 0)
    path = str(tmp_path / "fine.tq")
    fine.save(path)
    loaded = TieredQIndex.load(path)
    assert loaded.fast_prefix_bits == BITS_PER_DIM
    for actual, reference in zip(loaded.search(queries, K), (scores, ids)):
        np.testing.assert_array_equal(actual, reference)


def test_full_mode_rejects_forced_fine_tiers():
    index = TieredQIndex(DIMENSIONS, BITS_PER_DIM * DIMENSIONS, mode="full", fine_tiers="on")
    with pytest.raises(RuntimeError, match="fine tiers"):
        index.add(steep_rows(ROWS, 26))
