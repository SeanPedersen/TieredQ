"""Unit checks for the #33 lossless code-length estimators."""
import math

import numpy as np

from entropy import code_bits, elias_fano_bits, fine_split_bits, kt_bits, tier_symbols

ROWS = 20000
rng = np.random.default_rng(0)


def pack(symbols, bits):
    """Rows of codes packed LSB-first at bits each, the codec row contract."""
    width = (symbols.shape[1] * bits + 7) // 8
    rows = np.zeros((symbols.shape[0], width + 3), np.uint8)
    for j in range(symbols.shape[1]):
        offset = j * bits
        window = symbols[:, j].astype(np.uint32) << (offset % 8)
        for byte in range(3):
            rows[:, offset // 8 + byte] |= ((window >> (8 * byte)) & 0xFF).astype(np.uint8)
    return rows[:, :width]


def test_unpack_round_trips_wide_codes():
    symbols = rng.integers(0, 1 << 12, (ROWS, 7))
    tier = np.array([0, 7, 12, 0, 0])
    assert np.array_equal(tier_symbols(pack(symbols, 12), tier, np.arange(7)), symbols)


def test_uniform_costs_about_log_alphabet():
    alphabet = 16
    counts = np.bincount(rng.integers(0, alphabet, ROWS), minlength=alphabet)
    assert abs(kt_bits(counts, alphabet) / ROWS - math.log2(alphabet)) < 0.01


def test_degenerate_costs_about_nothing():
    counts = np.zeros(16, int)
    counts[3] = ROWS
    assert kt_bits(counts, 16) / ROWS < 0.01


def test_two_bit_gaussian_lloyd_max_entropy():
    probabilities = [0.1631, 0.3369, 0.3369, 0.1631]  # Lloyd-Max 2-bit cells of N(0, 1)
    symbols = rng.choice(4, (ROWS, 1), p=probabilities)
    estimate = code_bits(pack(symbols, 2), [np.array([0, 1, 2, 0, 0])]) / ROWS
    assert abs(estimate - 1.911) < 0.02


def test_elias_fano_formula():
    assert elias_fano_bits([0, 4], 64) == 4 * (2 + 4)


def test_fine_split_partitions_code_bits():
    symbols = rng.integers(0, 1 << 10, (ROWS, 3))
    split = fine_split_bits(pack(symbols, 10), [np.array([0, 3, 10, 0, 0])], 6)
    assert split["prefix_raw"] == ROWS * 3 * 6 and split["fine_raw"] == ROWS * 3 * 4
    assert split["fine_given_prefix"] >= split["fine_coded"] - 1e-6 * ROWS  # independent fine bits gain no context
