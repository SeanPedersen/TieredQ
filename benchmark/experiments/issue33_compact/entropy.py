"""Ideal lossless code lengths for TieredQ codec rows (#33): what a compact storage format could save, losslessly.

Every estimate is an adaptive Krichevsky-Trofimov code length, so model cost is included, and wide alphabets (12-bit
fine codes, fp16 metadata) on 100K rows are not over-credited the way plug-in entropy would be.
"""
from functools import lru_cache
import math

import numpy as np

METADATA_BYTES = 5  # centroid byte, fp16 factor, fp16 correction (src/tieredq_layout.h Layout)
ID_BITS = 32
CENTROID_BITS = 8
HALF_BITS = 16
COLUMN_CHUNK = 128  # bounds unpacking memory to N x 128 codes
WINDOW_BYTES = 3  # a code of up to 12 bits at any bit offset spans at most three bytes


@lru_cache(maxsize=None)
def _lgamma_table(size, shift):
    """lgamma(c + shift) for c in [0, size), so per-count terms are one gather instead of a Python call each."""
    return np.array([math.lgamma(c + shift) for c in range(size)])


def kt_bits(counts, alphabet):
    """KT code length in bits of the symbol counts of each context (counts: [..., alphabet])."""
    counts = np.asarray(counts).reshape(-1, alphabet)
    totals = counts.sum(axis=1)
    totals = totals[totals > 0]
    nonzero = counts[counts > 0]
    size = int(totals.max(initial=0)) + 1
    nats = (np.sum(_lgamma_table(size, alphabet / 2)[totals]) - totals.size * math.lgamma(alphabet / 2)
            - np.sum(_lgamma_table(size, 0.5)[nonzero]) + nonzero.size * math.lgamma(0.5))
    return float(nats / math.log(2))


def tier_symbols(rows, tier, columns):
    """Codes [N, len(columns)] of the given coordinates of one tier (start, dimensions, bits, scale_group, offset)."""
    bits, bit_offset = int(tier[2]), int(tier[4])
    padded = np.pad(rows, ((0, 0), (0, WINDOW_BYTES)))
    offsets = bit_offset + bits * np.asarray(columns)
    first = offsets // 8
    window = sum(padded[:, first + byte].astype(np.uint32) << (8 * byte) for byte in range(WINDOW_BYTES))
    return (window >> (offsets % 8).astype(np.uint32)) & ((1 << bits) - 1)


def rank_symbols(rows, tiers):
    """(bits, codes [N, <= COLUMN_CHUNK]) of every rank, a chunk of one tier at a time."""
    for tier in tiers:
        for first in range(0, int(tier[1]), COLUMN_CHUNK):
            columns = np.arange(first, min(first + COLUMN_CHUNK, int(tier[1])))
            yield int(tier[2]), tier_symbols(rows, tier, columns).astype(np.int64)


def column_bits(symbols, alphabet, contexts=None, context_count=1):
    """KT bits of each column of symbols coded with its own model (per context when contexts is given)."""
    keys = symbols if contexts is None else contexts * alphabet + symbols
    return sum(kt_bits(np.bincount(column, minlength=alphabet * context_count), alphabet) for column in keys.T)


def code_bits(rows, tiers, contexts=None, context_count=1):
    """Total KT bits of all codes, each rank coded with its own model (per context when contexts is given)."""
    context = None if contexts is None else contexts[:, None].astype(np.int64)
    return sum(column_bits(symbols, 1 << bits, context, context_count) for bits, symbols in rank_symbols(rows, tiers))


def fine_split_bits(rows, tiers, prefix_width):
    """Raw and KT bits of the scanned prefixes and of the fine side rows (#26 split: prefix = code >> fine bits)."""
    out = dict(prefix_raw=0.0, prefix_coded=0.0, fine_raw=0.0, fine_coded=0.0, fine_given_prefix=0.0)
    for bits, symbols in rank_symbols(rows, tiers):
        prefix_bits = min(bits, prefix_width)
        fine_bits = bits - prefix_bits
        prefix = symbols >> fine_bits
        out["prefix_raw"] += prefix.size * prefix_bits
        out["prefix_coded"] += column_bits(prefix, 1 << prefix_bits)
        if not fine_bits:
            continue
        fine = symbols & ((1 << fine_bits) - 1)
        out["fine_raw"] += fine.size * fine_bits
        out["fine_coded"] += column_bits(fine, 1 << fine_bits)
        for prefix_column, fine_column in zip(prefix.T, fine.T):
            out["fine_given_prefix"] += column_bits(fine_column[:, None], 1 << fine_bits, prefix_column[:, None],
                                                    1 << prefix_bits)
    return out


def metadata_fields(rows):
    """Centroid ids, factor halves and correction halves of codec rows."""
    meta = rows[:, -METADATA_BYTES:]
    halves = meta[:, 1:].copy().view('<u2')
    return meta[:, 0], halves[:, 0], halves[:, 1]


def half_bits(values):
    """Cheaper of order-0 coding the fp16 values whole or as high byte plus high-conditioned low byte."""
    byte = 1 << 8
    whole = kt_bits(np.bincount(values, minlength=1 << HALF_BITS), 1 << HALF_BITS)
    high, low = values >> 8, values & (byte - 1)
    split = kt_bits(np.bincount(high, minlength=byte), byte) + kt_bits(
        np.bincount(high.astype(np.int64) * byte + low, minlength=byte * byte), byte)
    return min(whole, split)


def elias_fano_bits(list_sizes, universe):
    """Elias-Fano bits of ascending ids per list: n (2 + ceil(log2(universe / n))) for each non-empty list."""
    sizes = np.asarray(list_sizes, dtype=float)
    sizes = sizes[sizes > 0]
    return float(np.sum(sizes * (2 + np.ceil(np.log2(universe / sizes)))))
