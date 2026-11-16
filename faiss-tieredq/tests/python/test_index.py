"""faiss_tieredq.IndexIVFTieredQ from Python: training, search order per metric, ids, persistence, removal and a fixed
partition (set_centroids)."""
import numpy as np
import pytest

from faiss_tieredq import IndexIVFTieredQ

DIMENSIONS, LISTS, ROWS, QUERIES, K = 32, 8, 2000, 10, 10


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return (rng.standard_normal((ROWS, DIMENSIONS), dtype=np.float32),
            rng.standard_normal((QUERIES, DIMENSIONS), dtype=np.float32))


def build(rows, metric, quantizer="flat"):
    index = IndexIVFTieredQ(DIMENSIONS, LISTS, 2 * DIMENSIONS, metric=metric, quantizer=quantizer)
    index.train(rows)
    index.add(rows, ids=np.arange(ROWS, dtype=np.int64) * 3)
    return index


@pytest.mark.parametrize("metric", ["ip", "cosine", "l2"])
def test_search_orders_by_metric(data, metric):
    rows, queries = data
    index = build(rows, metric)
    assert index.is_trained and index.ntotal == ROWS and index.nlist == LISTS
    distances, labels = index.search(queries, k=K, nprobe=LISTS)
    assert distances.shape == labels.shape == (QUERIES, K) and labels.dtype == np.int64
    assert np.all(labels % 3 == 0)
    ordered = np.diff(distances, axis=1)
    assert np.all(ordered >= 0) if metric == "l2" else np.all(ordered <= 0)
    # Full-probe top hits of an approximate code still find most exact neighbors.
    if metric == "l2":
        truth = np.argsort(((queries[:, None, :] - rows[None]) ** 2).sum(-1), axis=1)[:, :K] * 3
    else:
        scale = np.linalg.norm(rows, axis=1) if metric == "cosine" else 1
        truth = np.argsort(-(queries @ rows.T) / scale, axis=1)[:, :K] * 3
    overlap = np.mean([len(np.intersect1d(a, b)) / K for a, b in zip(labels, truth)])
    assert overlap > 0.5


def test_save_load_and_remove(data, tmp_path):
    rows, queries = data
    index = build(rows, "ip", quantizer="hnsw_fp16")
    before = index.search(queries, k=K, nprobe=4, ef_search=32)
    path = str(tmp_path / "index.tqf")
    index.save(path)
    loaded = IndexIVFTieredQ.load(path)
    after = loaded.search(queries, k=K, nprobe=4, ef_search=32)
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_array_equal(before[0], after[0])
    removed = before[1][:, 0].copy()
    assert loaded.remove_ids(np.unique(removed)) == len(np.unique(removed))
    assert not np.isin(loaded.search(queries, k=K, nprobe=4, ef_search=32)[1], removed).any()
    assert loaded.storage_bytes > 0 and loaded.code_size > 0


def test_set_centroids_fixes_partition(data):
    rows, queries = data
    centroids = rows[:LISTS].copy()
    index = IndexIVFTieredQ(DIMENSIONS, LISTS, 2 * DIMENSIONS, metric="l2", quantizer="fp16")
    index.set_centroids(centroids)
    index.train(rows)
    index.add(rows)
    distances, labels = index.search(queries, k=K, nprobe=LISTS)
    assert (labels >= 0).all() and np.all(np.diff(distances, axis=1) >= 0)
    with pytest.raises(ValueError):
        index.set_centroids(centroids)
