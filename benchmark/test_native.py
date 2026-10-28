"""Native contracts on synthetic data, including nonmultiple batch and storage padding."""
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import codec_native


class NativeTests(unittest.TestCase):
    def test_batched_search_matches_serial_and_storage_exists(self):
        rng = np.random.default_rng(10)
        base = rng.normal(size=(129, 128)).astype(np.float32)
        probes = rng.normal(size=(35, 128)).astype(np.float32)
        os.environ["BENCH_SEED"] = "0"
        codec_native.set_threads(8)
        for codec in (True, False):
            for bits in (1,2,4,8):
                index = codec_native.Index(base, bits, 0, codec, False)
                batch = np.empty((35, 10), dtype=np.uint32)
                serial = np.empty_like(batch)
                index.search(probes, batch, 8)
                for i in range(len(probes)):
                    index.search(probes[i:i+1], serial[i:i+1], 1)
                np.testing.assert_array_equal(batch, serial)
                self.assertTrue(np.all(batch < len(base)))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/"index.bin"
                    size = index.storage(str(path))
                    self.assertGreaterEqual(size, path.stat().st_size)
                    self.assertGreater(size, len(base)*128*bits/8)

    def test_rabitq_native_l2_uses_original_norms(self):
        from data import ground_truth
        rng = np.random.default_rng(13)
        base = rng.normal(size=(1024, 128)).astype(np.float32)
        base *= np.linspace(.5, 2, len(base), dtype=np.float32)[:, None]
        queries = rng.normal(size=(32, 128)).astype(np.float32)
        truth = ground_truth(base, queries, True)
        os.environ["BENCH_SEED"] = "0"
        index = codec_native.Index(base, 8, 0, False, True)
        ids = np.empty((len(queries), 10), dtype=np.uint32)
        index.search(queries, ids, 8)
        recall = np.mean([len(set(a) & set(b))/10 for a,b in zip(truth,ids)])
        self.assertGreater(recall, .9)

    def test_flat_storage_has_no_per_vector_id_array(self):
        os.environ["BENCH_SEED"] = "0"
        rng = np.random.default_rng(31)
        dimensions, smaller_n, larger_n = 128, 1024, 2048
        base = rng.normal(size=(larger_n,dimensions)).astype(np.float32)
        sizes = []
        with tempfile.TemporaryDirectory() as directory:
            for n in (smaller_n,larger_n):
                index = codec_native.Index(base[:n],1,0,False,True)
                sizes.append(index.storage(str(Path(directory)/f"{n}.rq")))
        native_bytes_per_vector = dimensions//8 + 3*np.dtype(np.float32).itemsize
        self.assertEqual(sizes[1]-sizes[0],(larger_n-smaller_n)*native_bytes_per_vector)

    def test_saved_index_reloads_with_identical_results(self):
        os.environ["BENCH_SEED"] = "0"
        rng = np.random.default_rng(37)
        base = rng.normal(size=(300, 64)).astype(np.float32)
        probes = rng.normal(size=(20, 64)).astype(np.float32)
        for codec in (True, False):
            for bits in (1, 4):
                index = codec_native.Index(base, bits, 0, codec, False)
                expected = np.empty((len(probes), 10), dtype=np.uint32)
                index.search(probes, expected, 4)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/"index.bin"
                    self.assertEqual(index.save(str(path)), path.stat().st_size)
                    loaded = codec_native.Index.load(str(path), codec)
                actual = np.empty_like(expected)
                loaded.search(probes, actual, 4)
                np.testing.assert_array_equal(actual, expected)
                self.assertGreater(loaded.resident_bytes(), len(base)*64*bits/8)

if __name__ == "__main__":
    unittest.main()
