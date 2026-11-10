"""IVF router and coarse-training bindings: an exhaustive router equals exact ranking, the router persists, restricted
assignment over every super equals exact assignment, and invalid uses are rejected."""
import numpy as np
import pytest

from tieredq import TieredQIVFIndex

DIMENSIONS = 32
ROWS = 6000
LISTS = 512
LISTS_PER_SUPER = 16
NPROBE = 24
K = 10


def gaussian(rows, seed):
    decay = np.exp(-np.arange(DIMENSIONS) / DIMENSIONS)
    return (np.random.default_rng(seed).normal(size=(rows, DIMENSIONS)) * decay).astype(np.float32)


@pytest.fixture(scope="module")
def data():
    return gaussian(ROWS, 1), gaussian(32, 2)


def built(vectors, **training):
    index = TieredQIVFIndex(DIMENSIONS, 2 * DIMENSIONS, lists=LISTS, metric="l2", max_scan_bits=4)
    if training:
        index.set_coarse_training(**training)
    index.build(vectors)
    return index


def test_exhaustive_router_equals_exact_ranking(data):
    vectors, queries = data
    index = built(vectors)
    assert not index.has_router
    index.build_router(lists_per_super=LISTS_PER_SUPER, rerank=LISTS, min_super_probes=LISTS)
    assert index.has_router
    routed = index.search(queries, K, nprobe=NPROBE, routing="router")
    exact = index.search(queries, K, nprobe=NPROBE, routing="exact")
    np.testing.assert_array_equal(routed[1], exact[1])
    np.testing.assert_array_equal(routed[0], exact[0])


def test_router_persists(data, tmp_path):
    vectors, queries = data
    index = built(vectors)
    index.build_router(lists_per_super=LISTS_PER_SUPER, min_super_probes=2, rerank=32)
    expected = index.search(queries, K, nprobe=NPROBE, routing="router", super_probes=3)
    index.save(tmp_path / "routed.tq")
    loaded = TieredQIVFIndex.load(tmp_path / "routed.tq")
    assert loaded.has_router
    actual = loaded.search(queries, K, nprobe=NPROBE, routing="router", super_probes=3)
    np.testing.assert_array_equal(actual[1], expected[1])


def test_restricted_assignment_over_every_super_is_exact(data):
    vectors, _ = data
    exact = built(vectors).list_assignment()
    restricted = built(vectors, assignment_supers=16, assignment_probes=16).list_assignment()
    np.testing.assert_array_equal(restricted, exact)


def test_invalid_router_and_training_uses(data):
    vectors, queries = data
    index = built(vectors)
    with pytest.raises(RuntimeError):
        index.search(queries, K, nprobe=NPROBE, routing="router")
    with pytest.raises(ValueError):
        index.search(queries, K, nprobe=NPROBE, routing="nearest")
    with pytest.raises(RuntimeError):
        TieredQIVFIndex(DIMENSIONS, 2 * DIMENSIONS, lists=LISTS).build_router()
    with pytest.raises(ValueError):
        TieredQIVFIndex(DIMENSIONS, 2 * DIMENSIONS).set_coarse_training(iterations=5)


def test_probe_lists_match_exhaustive_router(data):
    vectors, queries = data
    index = built(vectors)
    index.build_router(lists_per_super=LISTS_PER_SUPER, rerank=LISTS, min_super_probes=LISTS)
    for query in queries[:8]:
        routed = index.probe_lists(query, nprobe=NPROBE, routing="router")
        assert routed.shape == (NPROBE,)
        np.testing.assert_array_equal(routed, index.probe_lists(query, nprobe=NPROBE, routing="exact"))


def test_set_lists_fixes_the_partition(data):
    vectors, _ = data
    centroids = gaussian(LISTS, 9)
    index = TieredQIVFIndex(DIMENSIONS, 2 * DIMENSIONS, lists=LISTS, metric="l2", max_scan_bits=4)
    index.set_lists(centroids)
    index.build(vectors)
    nearest = ((centroids ** 2).sum(1)[None, :] - 2 * vectors @ centroids.T).argmin(1)
    assert (index.list_assignment() == nearest).mean() > 0.99
    with pytest.raises(ValueError):
        index.set_lists(centroids[:-1])
