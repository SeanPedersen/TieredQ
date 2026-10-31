"""Tests for shift_data.py: cluster split, query grouping and the written eval directories."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import shift_data


def blobs(rows_per_blob, blobs_count, d, rng):
    centers = rng.normal(size=(blobs_count, d)) * 10
    rows = np.concatenate([center + rng.normal(size=(rows_per_blob, d)) for center in centers]).astype(np.float32)
    return rows, np.repeat(np.arange(blobs_count), rows_per_blob)


class ShiftDataTest(unittest.TestCase):
    def test_kmeans_recovers_separated_blobs(self):
        rng = np.random.default_rng(1)
        rows, truth_labels = blobs(50, 4, 8, rng)
        labels = shift_data.assign(rows, shift_data.kmeans(rows, 4, 10, np.random.default_rng(2)))
        for blob in range(4):
            self.assertEqual(len(set(labels[truth_labels == blob])), 1)

    def test_heldout_queries_need_a_majority(self):
        row_heldout = np.array([True] * 6 + [False] * 14)
        truth = np.array([list(range(10)), list(range(10, 20)), [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]])
        self.assertEqual(shift_data.heldout_queries(truth, row_heldout).tolist(), [True, False, False])

    def test_main_writes_partitioned_query_sets(self):
        rng = np.random.default_rng(3)
        d = 8
        base, _ = blobs(40, 8, d, rng)
        queries = base[::16] + 0.01
        truth = np.argsort(-(queries @ base.T), axis=1)[:, :10].astype(np.uint32)
        with tempfile.TemporaryDirectory() as directory:
            data, out = Path(directory) / "data", Path(directory) / "out"
            data.mkdir()
            base.tofile(data / "base.f32"); queries.astype(np.float32).tofile(data / "queries.f32")
            truth.tofile(data / "truth.u32")
            shift_data.main(data, d, out, 8, 0.25)
            manifest = json.loads((out / "manifest.json").read_text())
            excluded = np.fromfile(out / "excluded_rows.u32", dtype=np.uint32)
            self.assertEqual(len(manifest["heldout_clusters"]), 2)
            self.assertEqual(len(excluded), manifest["heldout_rows"])
            counts = [len(np.fromfile(out / part / "truth.u32", dtype=np.uint32)) // 10 for part in ("heldout", "seen")]
            self.assertEqual(counts, [manifest["heldout_queries"], len(queries) - manifest["heldout_queries"]])
            self.assertTrue((out / "seen" / "base.f32").is_symlink())


if __name__ == "__main__":
    unittest.main()
