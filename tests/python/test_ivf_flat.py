"""Unquantized IVF-Flat binding: knn_graph probing every list is the exhaustive self-excluding k-NN graph, it equals
search() of the rows at any nprobe, scores follow the metric, and invalid input is rejected."""
import numpy as np
import pytest

from tieredq import TieredQIVFFlatIndex

DIMENSIONS = 16
ROWS = 800
LISTS = 8
K = 10
PADDING_ID = np.iinfo(np.uint32).max


def rows(seed=0):
    return np.random.default_rng(seed).normal(size=(ROWS, DIMENSIONS)).astype(np.float32)


def exhaustive_graph(data):
    distances = ((data[:, None, :] - data[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1)[:, :K]


def built(data, metric="l2"):
    index = TieredQIVFFlatIndex(DIMENSIONS, lists=LISTS, metric=metric)
    index.build(data)
    return index


def test_full_probe_graph_is_exhaustive():
    data = rows()
    scores, ids = built(data).knn_graph(K, nprobe=LISTS)
    assert ids.shape == (ROWS, K) and scores.dtype == np.float32
    truth = exhaustive_graph(data)
    hits = sum(len(np.intersect1d(a, b)) for a, b in zip(ids, truth))
    assert hits >= ROWS * K - ROWS // 100
    first = ((data - data[ids[:, 0]]) ** 2).sum(1)
    np.testing.assert_allclose(scores[:, 0], -first, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("nprobe", [1, 3])
def test_graph_equals_search_without_self(nprobe):
    data = rows(1)
    index = built(data)
    _, graph = index.knn_graph(K, nprobe=nprobe)
    _, searched = index.search(data, K + 1, nprobe=nprobe)
    for row, (found, hits) in enumerate(zip(graph, searched)):
        np.testing.assert_array_equal(found, hits[hits != row][:K])


@pytest.mark.parametrize("metric", ["ip", "cosine"])
def test_metric_scores(metric):
    data = rows(2)
    scores, ids = built(data, metric).knn_graph(K, nprobe=LISTS)
    top = data[ids[:, 0]]
    expected = (data * top).sum(1)
    if metric == "cosine":
        expected /= np.linalg.norm(data, axis=1) * np.linalg.norm(top, axis=1)
    np.testing.assert_allclose(scores[:, 0], expected, rtol=1e-3, atol=1e-3)


def test_shape_padding_and_validation():
    data = rows(3)[:6]
    index = TieredQIVFFlatIndex(DIMENSIONS, lists=2)
    index.build(data)
    assert len(index) == 6 and index.lists == 2 and index.list_assignment().shape == (6,)
    assert set(index.build_stats) == {"train_s", "assign_s", "store_s"}
    _, ids = index.knn_graph(K, nprobe=2)
    assert (ids[:, :5] != PADDING_ID).all() and (ids[:, 5:] == PADDING_ID).all()
    _, single = index.search(data[0], 3, nprobe=2)
    assert single.shape == (3,) and single[0] == 0
    assert TieredQIVFFlatIndex(DIMENSIONS).lists is None
    with pytest.raises(ValueError):
        index.build(np.zeros((4, DIMENSIONS + 1), np.float32))
    with pytest.raises(ValueError):
        TieredQIVFFlatIndex(DIMENSIONS, lists=0)
    with pytest.raises(ValueError):
        TieredQIVFFlatIndex(DIMENSIONS, metric="hamming")
    with pytest.raises(RuntimeError):
        TieredQIVFFlatIndex(DIMENSIONS, lists=4).build(data[:2])
