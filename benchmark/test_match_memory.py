import unittest

from match_memory import MAX_STEP_DOWNS, METRICS, RESOLUTION, companion_delta, confirm, recall_summaries

PEAK = METRICS["peak"]
DIMENSIONS = 200
TARGET = 335.0
NOISY_SEED = 1
SEED_NOISE = 12.0


class FakeCells:
    """Peak memory grows with code_bits; one seed reads SEED_NOISE bits/vector higher, like RSS noise."""

    def __init__(self):
        self.measured = []

    def measure(self, dataset, codec, bits, seed, code_bits=0):
        self.measured.append((code_bits, seed))
        peak = 140 + code_bits + (SEED_NOISE if seed == NOISY_SEED else 0)
        return {PEAK: str(peak), "code_bits": str(code_bits), "seed": str(seed)}


def run_confirm(cells, candidates):
    return confirm(cells, "glove", "peak", TARGET, candidates, DIMENSIONS, [0, 1, 2], lambda entry: None, {})


class ConfirmTest(unittest.TestCase):
    def test_steps_below_cached_budgets_when_seed_noise_overshoots(self):
        cells = FakeCells()
        chosen, rows = run_confirm(cells, [190])
        self.assertEqual(chosen, 190 - RESOLUTION)
        self.assertTrue(all(float(row[PEAK]) <= TARGET for row in rows))

    def test_gives_up_after_bounded_step_downs(self):
        cells = FakeCells()
        chosen, rows = run_confirm(cells, [400])
        self.assertIsNone(chosen)
        self.assertEqual(rows, [])
        self.assertEqual(min(code for code, _ in cells.measured), 400 - MAX_STEP_DOWNS * RESOLUTION)

    def test_no_candidates_measures_nothing(self):
        cells = FakeCells()
        self.assertEqual(run_confirm(cells, []), (None, []))
        self.assertEqual(cells.measured, [])


class CompanionRecallTest(unittest.TestCase):
    def test_summarises_companion_recall_when_every_cell_has_it(self):
        rows = [{"recall10": "0.9", "companion_recall10": "0.8"}, {"recall10": "0.7", "companion_recall10": "0.6"}]
        recalls = recall_summaries(rows)
        self.assertAlmostEqual(recalls["recall10"]["mean"], 0.8)
        self.assertAlmostEqual(recalls["companion_recall10"]["mean"], 0.7)

    def test_omits_companion_recall_for_cells_without_it(self):
        recalls = recall_summaries([{"recall10": "0.9", "companion_recall10": ""}])
        self.assertNotIn("companion_recall10", recalls)
        self.assertEqual(companion_delta(recalls, recalls), "–")

    def test_companion_delta_is_tieredq_minus_reference(self):
        reference = recall_summaries([{"recall10": "0.9", "companion_recall10": "0.80"}])
        tieredq = recall_summaries([{"recall10": "0.9", "companion_recall10": "0.85"}])
        self.assertEqual(companion_delta(reference, tieredq), "+0.0500")


if __name__ == "__main__":
    unittest.main()
