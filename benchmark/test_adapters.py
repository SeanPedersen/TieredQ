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

    def test_exact_code_budget_sets_tieredq_tiers(self):
        rng = np.random.default_rng(41)
        base = rng.normal(size=(600, 64)).astype(np.float32)
        sizes = []
        with tempfile.TemporaryDirectory() as directory:
            for code_bits in (40, 100):
                index = codec_native.Index(base, 2, 0, True, False, True, code_bits)
                tiers = index.diagnostics()["tiers"]
                # Fast nibble models bound padded storage (#9): up to an unpaired 2-bit and three 1-bit slots more.
                realised = sum(tier["dimensions"]*tier["bits"] for tier in tiers)
                self.assertLessEqual(realised, -(-(code_bits + 5) // 8) * 8)
                sizes.append(index.save(str(Path(directory)/f"{code_bits}.bin")))
        self.assertLess(sizes[0], sizes[1])

    def test_native_l2_metric_ranks_like_exact_l2(self):
        rng = np.random.default_rng(43)
        base = (rng.normal(size=(800, 32)) * rng.uniform(0.2, 3.0, size=(800, 1)) + 1.5).astype(np.float32)
        probes = base[:20] + 0.01*rng.normal(size=(20, 32)).astype(np.float32)
        index = codec_native.Index(base, 4, 0, True, True, True, 0, True)
        ids = np.empty((20, 10), dtype=np.uint32)
        index.search(probes, ids, 1)
        self.assertEqual(sum(tier["dimensions"]*tier["bits"] for tier in index.diagnostics()["tiers"]) > 0, True)
        exact = np.argsort(((probes[:, None, :]-base[None])**2).sum(-1), axis=1)[:, :10]
        overlap = np.mean([len(set(a) & set(b)) for a, b in zip(ids, exact)])
        self.assertGreaterEqual(overlap, 8)

    def test_tieredq_variants_search_deeper_than_ten(self):
        from data import ground_truth
        from measure import recalls
        rng = np.random.default_rng(47)
        base = rng.normal(size=(2048, 64)).astype(np.float32)
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        probes = base[:16] + 0.05*rng.normal(size=(16, 64)).astype(np.float32)
        truth = ground_truth(base, probes, False, 100)
        codec_native.set_threads(8)
        for variant in (dict(), dict(fine_tiers=False), dict(exact=True)):
            index = codec_native.Index(base, 8, 0, True, False, True, 0, True, **variant)
            ids = np.empty((16, 100), dtype=np.uint32)
            index.search(probes, ids, 8)
            serial = np.empty_like(ids)
            for i in range(len(probes)):
                index.search(probes[i:i+1], serial[i:i+1], 1)
            np.testing.assert_array_equal(ids, serial)
            self.assertGreater(recalls(truth, ids)["recall100"], .9, variant)
        rabitq = codec_native.Index(base, 4, 0, False, False)
        ids = np.empty((16, 100), dtype=np.uint32)
        rabitq.search(probes, ids, 8)
        self.assertGreater(recalls(truth, ids)["recall100"], .8)

    def test_exact_scores_need_the_fast_index(self):
        base = np.random.default_rng(53).normal(size=(64, 16)).astype(np.float32)
        with self.assertRaises(ValueError):
            codec_native.Index(base, 4, 0, True, False, False, exact=True)

if __name__ == "__main__":
    unittest.main()
