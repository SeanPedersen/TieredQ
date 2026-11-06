"""Adapter contracts plus Qdrant TQ+ save/reload smoke tests before long jobs."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

B = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(B))
import numpy as np
from qdrant_codec import QdrantIndex


class QdrantSmoke(unittest.TestCase):
    def test_reload_preserves_top10(self):
        rng = np.random.default_rng(4)
        base = rng.normal(size=(1024, 128)).astype(np.float32)
        queries = rng.normal(size=(35, 128)).astype(np.float32)
        for bits in (1, 2, 4):
            for l2 in (False, True):
                index = QdrantIndex.build(base, bits, l2)
                expected = np.empty((len(queries), 10), dtype=np.uint32)
                index.search(queries, expected, 8)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp)/'index'
                    disk = index.save(path)
                    loaded = QdrantIndex.load(path)
                    actual = np.empty_like(expected)
                    loaded.search(queries, actual, 8)
                    np.testing.assert_array_equal(expected, actual)
                    self.assertEqual(disk, sum(p.stat().st_size for p in path.iterdir()))
                del index, loaded


if __name__ == '__main__':
    os.environ['BENCH_SEED'] = '0'
    loader = unittest.defaultTestLoader
    suite = loader.loadTestsFromNames(['test_native', 'test_reference', 'test_benchmark', 'test_match_memory'])
    suite.addTests(loader.loadTestsFromTestCase(QdrantSmoke))
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        sys.exit(1)
