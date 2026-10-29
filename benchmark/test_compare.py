"""Guard comparisons against matching nominal bits instead of measured storage."""
import unittest
from compare import compare, operating_points


class ComparisonTests(unittest.TestCase):
    def test_batch_only_run_does_not_invent_single_query_latency(self):
        row = dict(dataset="sift", threads=8, codec="tieredq", batch=32, target_bits=8,
                   seed=0, recall10=.99, effective_bytes_per_vector=142, qps=200,
                   build_s=2, p50_ms=160, p95_ms=180, encode_vps=1000)
        point = operating_points([row])[0]["points"][0]
        self.assertEqual(point["qps_seed0"], 200)
        self.assertIsNone(point["p50_ms_seed0"])
        self.assertIsNone(point["p95_ms_seed0"])

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

    def test_every_reference_codec_is_an_anchor(self):
        points = [dict(dataset="gemini", threads=8, codec="qdrant", points=[dict(bytes_per_vector=40, recall_mean=.7)]),
                  dict(dataset="gemini", threads=8, codec="rabitq", points=[dict(bytes_per_vector=42, recall_mean=.72)]),
                  dict(dataset="gemini", threads=8, codec="tieredq", points=[dict(bytes_per_vector=39, recall_mean=.74, qps_seed0=900)])]
        self.assertEqual([r["anchor_codec"] for r in compare(points)], ["qdrant", "rabitq"])


if __name__ == "__main__":
    unittest.main()
