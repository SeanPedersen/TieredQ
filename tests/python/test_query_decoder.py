"""The optional query decoder changes search scores and is not stored in the index."""
import re

import numpy as np
import pytest

from tieredq import TieredQIVFIndex, TieredQIndex, TieredQNestedIVFIndex


def _rows(count, dimensions, seed):
    rng = np.random.default_rng(seed)
    rows = rng.standard_normal((count, dimensions)).astype(np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _shape(index):
    empty = np.zeros((1, 1), np.float32)
    with pytest.raises(RuntimeError, match="expected basis") as caught:
        index.set_query_decoder(empty, empty, empty, np.zeros(1, np.float32), np.zeros(1, np.float32))
    match = re.search(r"expected basis \((\d+), (\d+)\), centroids \((\d+),", str(caught.value))
    assert match is not None
    return tuple(int(part) for part in match.groups())


def _decoder(index, scale):
    dimensions, coded, centroids = _shape(index)
    basis = np.zeros((dimensions, coded), np.float32)
    basis[0, 0] = 1
    centers = np.zeros((centroids, dimensions), np.float32)
    centers[:, 0] = 1
    return basis, centers, centers.copy(), np.full(coded, scale, np.float32), np.zeros(coded, np.float32)


def _search(index, query):
    if isinstance(index, (TieredQIVFIndex, TieredQNestedIVFIndex)):
        return index.search(query, k=5, nprobe=4)
    return index.search(query, k=5)


def _assert_optional(index, query, path=None):
    original = _search(index, query)
    index.set_query_decoder(*_decoder(index, scale=3))
    shifted = _search(index, query)
    assert not np.array_equal(shifted[0], original[0])
    index.clear_query_decoder()
    restored = _search(index, query)
    np.testing.assert_array_equal(restored[0], original[0])
    np.testing.assert_array_equal(restored[1], original[1])
    if path is None:
        return
    index.set_query_decoder(*_decoder(index, scale=3))
    index.save(str(path))
    loaded = type(index).load(str(path))
    reloaded = _search(loaded, query)
    np.testing.assert_array_equal(reloaded[0], original[0])
    np.testing.assert_array_equal(reloaded[1], original[1])


def test_flat_and_ivf_decoder_is_cleared_and_not_saved(tmp_path):
    rows = _rows(64, 32, seed=0)
    query = rows[:1]
    flat = TieredQIndex(dimensions=32, code_bits=32, fine_tiers=False)
    flat.add(rows)
    ivf = TieredQIVFIndex(dimensions=32, code_bits=32, lists=4, fine_tiers=False)
    ivf.build(rows)
    _assert_optional(flat, query, tmp_path / "flat.tq")
    _assert_optional(ivf, query, tmp_path / "ivf.tq")


def test_nested_decoder_changes_the_prefix_scan_and_clears(tmp_path):
    rows = _rows(48, 16, seed=1)
    index = TieredQNestedIVFIndex.build(rows, prefix_bits=16, total_bits=48, lists=4, fine_tiers=False)
    index.attach_tier(index.encode_tier(rows))
    _assert_optional(index, rows[:1])


def test_fine_tier_model_rejects_the_decoder():
    rows = _rows(64, 32, seed=2)
    index = TieredQIndex(dimensions=32, code_bits=8 * 32, fine_tiers=True)
    index.add(rows)
    if index.fast_prefix_bits == 0:
        pytest.skip("this budget did not grow a fine-tier prefix")
    basis = np.zeros((32, 1), np.float32)
    with pytest.raises(RuntimeError, match="fine-tier prefix"):
        index.set_query_decoder(basis, basis, basis, np.zeros(1, np.float32), np.zeros(1, np.float32))
