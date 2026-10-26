"""Check metric transformations, chunked truth and complete-call timing contracts."""
import unittest
import hashlib
from pathlib import Path
import tempfile
import numpy as np
from data import codec_vectors, ground_truth, download
from measure import measure
from l2_transform import L2Transform


class BenchmarkTests(unittest.TestCase):
    def test_chunked_truth_matches_direct_l2_and_cosine(self):
        rng = np.random.default_rng(5)
        base = rng.normal(size=(16389, 6)).astype(np.float32)
        queries = rng.normal(size=(35, 6)).astype(np.float32)
        for l2 in (False, True):
            scores = queries @ base.T
            if l2:
                scores = -np.sum((queries[:, None] - base[None])**2, axis=2)
            expected = np.argsort(-scores, axis=1)[:, :10]
            np.testing.assert_array_equal(ground_truth(base, queries, l2), expected)

    def test_l2_lift_preserves_exact_rankings(self):
        rng = np.random.default_rng(7)
        base = rng.normal(size=(40, 8)).astype(np.float32)
        queries = rng.normal(size=(4, 8)).astype(np.float32)
        vectors, probes = codec_vectors(base, queries, True)
        np.testing.assert_array_equal(ground_truth(base, queries, True), ground_truth(vectors, probes, False))

    def test_query_metric_lift_preserves_scores_and_times_the_query_transform(self):
        from measure import L2Search
        rng = np.random.default_rng(17)
        base = rng.normal(size=(128, 8)).astype(np.float32)
        queries = rng.normal(size=(7, 8)).astype(np.float32)
        transform = L2Transform.fit(base)
        vectors, probes = codec_vectors(base, queries, True)
        np.testing.assert_allclose(transform.base_vectors(base) @ transform.queries(queries).T,
                                   vectors @ probes.T, rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(ground_truth(base, queries, True),
                                     ground_truth(transform.base_vectors(base), transform.queries(queries), False))
        self.assertGreaterEqual(transform.storage_bytes, transform.base_map.nbytes + transform.query_map.nbytes)

        class Capture:
            def search(self, actual, output, threads):
                np.testing.assert_array_equal(actual, transform.queries(queries))
        L2Search(Capture(), transform.scale, transform).search(queries, None, 8)

    def test_query_metric_lift_handles_rank_deficient_calibration(self):
        queries = np.random.default_rng(23).normal(size=(7, 8)).astype(np.float32)
        for base in (np.zeros((16, 8), dtype=np.float32), np.ones((16, 8), dtype=np.float32)):
            transform = L2Transform.fit(base)
            self.assertTrue(np.isfinite(transform.query_map).all())
            vectors = transform.base_vectors(base)
            self.assertTrue(np.isfinite(vectors).all())
            scaled = base / transform.scale
            expected = queries / transform.scale @ scaled.T - np.sum(scaled * scaled, axis=1) / 2
            np.testing.assert_allclose(transform.queries(queries) @ vectors.T, expected, rtol=2e-5, atol=2e-6)

    def test_download_checks_cached_source_without_overwriting(self):
        payload = b"verified source"
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"source.hdf5"
            path.write_bytes(payload)
            download("https://example.invalid/not-used",path,expected)
            path.write_bytes(b"wrong source")
            with self.assertRaisesRegex(ValueError,"checksum mismatch"):
                download("https://example.invalid/not-used",path,expected)
            self.assertEqual(path.read_bytes(),b"wrong source")

    def test_batch_tail_and_recall(self):
        class Exact:
            def search(self, queries, output, threads):
                output[:] = np.arange(10)
        queries = np.ones((35, 2), dtype=np.float32)
        truth = np.tile(np.arange(10), (35, 1))
        result = measure(Exact(), queries, truth, 8, 32)
        self.assertEqual(result["recall10"], 1)
        self.assertGreater(result["qps"], 0)
        self.assertGreaterEqual(result["p95_ms"], result["p50_ms"])


if __name__ == "__main__":
    unittest.main()
