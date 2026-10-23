"""Guard comparisons against matching nominal bits instead of measured storage."""
import unittest
from compare import compare


class ComparisonTests(unittest.TestCase):
    def test_budget_filter_and_recall_floor(self):
        anchor = dict(bytes_per_vector=40, recall_mean=.9, qps_seed0=100)
        compact = dict(bytes_per_vector=30, recall_mean=.8, qps_seed0=300)
        accurate = dict(bytes_per_vector=60, recall_mean=.95, qps_seed0=200)
        points = [dict(dataset="glove", threads=1, codec="rabitq", points=[anchor]),
                  dict(dataset="glove", threads=1, codec="tieredq", points=[compact, accurate])]
        result = compare(points)[0]
        self.assertEqual(result["tieredq_best_recall_within_budget"], compact)
        self.assertEqual(result["tieredq_fastest_at_least_anchor_recall"], accurate)

    def test_no_matching_point_is_explicit(self):
        points = [dict(dataset="sift", threads=8, codec="rabitq", points=[dict(bytes_per_vector=10, recall_mean=.99)]),
                  dict(dataset="sift", threads=8, codec="tieredq", points=[dict(bytes_per_vector=20, recall_mean=.9, qps_seed0=500)])]
        result = compare(points)[0]
        self.assertIsNone(result["tieredq_best_recall_within_budget"])
        self.assertIsNone(result["tieredq_fastest_at_least_anchor_recall"])


if __name__ == "__main__":
    unittest.main()
