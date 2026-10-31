"""Tests for query_energy.py: tier parsing, rank bits, the greedy allocator and the error models."""
import unittest
import numpy as np
import query_energy as qe


class QueryEnergyTest(unittest.TestCase):
    def test_parse_tiers_and_rank_bits(self):
        layouts = qe.parse_tiers("nesting\tmodel\t512\tcode_bits 512\tbytes 69\ttiers 4x3 3x49\nother line\n")
        self.assertEqual(layouts, {512: [(4, 3), (3, 49)]})
        bits = qe.rank_bits(layouts[512], 60)
        self.assertEqual(bits[:3].tolist(), [4] * 3)
        self.assertEqual(int(bits.sum()), 4 * 3 + 3 * 49)
        self.assertEqual(int(np.count_nonzero(bits)), 52)

    def test_greedy_bits_spends_budget_on_heavy_ranks(self):
        bits = qe.greedy_bits(np.array([100.0, 10.0, 1.0, 0.0]), 6)
        self.assertEqual(int(bits.sum()), 6)
        self.assertEqual(bits[3], 0)
        self.assertTrue(bits[0] >= bits[1] >= bits[2])

    def test_greedy_bits_is_optimal_on_small_case(self):
        weights = np.array([5.0, 3.0, 1.0])
        best = min(qe.lloyd_max_error(weights, np.ones(3), np.array(b, dtype=float))
                   for b in np.ndindex(5, 5, 5) if sum(b) == 5)
        self.assertAlmostEqual(qe.lloyd_max_error(weights, np.ones(3), qe.greedy_bits(weights, 5)), best)

    def test_isotropic_error_matches_flat_spectrum(self):
        flat = np.ones(8)
        self.assertAlmostEqual(qe.isotropic_error(flat, flat, 1), qe.lloyd_max_error(flat, flat, np.ones(8)))

    def test_spectrum_orders_ranks_descending(self):
        directory = self.enterContext(__import__("tempfile").TemporaryDirectory())
        prefix = f"{directory}/pca"
        values = np.array([1.0, 2.0, 3.0])
        vectors = np.eye(3)
        values.tofile(f"{prefix}.eigenvalues.f64")
        vectors.T.copy().tofile(f"{prefix}.eigenvectors.f64")
        variance, basis = qe.load_spectrum(prefix, 3)
        self.assertEqual(variance.tolist(), [3.0, 2.0, 1.0])
        np.testing.assert_array_equal(basis[:, 0], [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
