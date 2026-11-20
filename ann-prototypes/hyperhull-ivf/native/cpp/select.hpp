// Fast selection of the n smallest scores (result unordered).
//
// A threshold is estimated from every kStride-th score (rank selection on the small sample, with slack),
// all scores at or below it are collected in one SIMD pass, and the exact n are then chosen among
// those ~1.5·n survivors. If the threshold was too tight (rare), fall back to selecting over
// everything. Rank selection is a bisection over order-preserving int32 keys (SIMD counting passes).
// Handles ±inf scores; scores must not be NaN.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace hh {

struct SelectScratch {
    std::vector<int32_t> sample, keys;
    std::vector<uint32_t> positions;
};

namespace select_detail {
constexpr size_t kStride = 16;
constexpr size_t kMinSampled = 512;  // below this, selecting over everything is cheapest
constexpr float kSlack = 1.5f;
constexpr size_t kSlackAbs = 8;
constexpr size_t kLanes = 8;

// For each 8-bit lane mask: the indices of its set lanes, packed to the front (AVX2 compress).
inline const std::array<std::array<uint32_t, kLanes>, 256>& compress_table() {
    static const auto table = [] {
        std::array<std::array<uint32_t, kLanes>, 256> t{};
        for (unsigned mask = 0; mask < 256; ++mask) {
            size_t k = 0;
            for (uint32_t lane = 0; lane < kLanes; ++lane)
                if (mask >> lane & 1) t[mask][k++] = lane;
        }
        return t;
    }();
    return table;
}

// Writes the positions i with scores[i] <= threshold to out (needs count + kLanes capacity); returns how many.
inline size_t positions_at_most(const float* scores, size_t count, float threshold, uint32_t* out) {
    const auto& table = compress_table();
    const __m256 limit = _mm256_set1_ps(threshold);
    const __m256i lane_ids = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    size_t kept = 0, i = 0;
    for (; i + kLanes <= count; i += kLanes) {
        const unsigned mask = static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(scores + i), limit, _CMP_LE_OQ)));
        const __m256i lanes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(table[mask].data()));
        const __m256i positions = _mm256_add_epi32(_mm256_permutevar8x32_epi32(lane_ids, lanes), _mm256_set1_epi32(static_cast<int>(i)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + kept), positions);
        kept += static_cast<size_t>(std::popcount(mask));
    }
    for (; i < count; ++i) {
        out[kept] = static_cast<uint32_t>(i);
        kept += scores[i] <= threshold;
    }
    return kept;
}
// Order-preserving int32 key of a float (an involution): a < b ⇔ key(a) < key(b) for non-NaN floats.
inline int32_t float_key(float f) {
    const auto bits = std::bit_cast<int32_t>(f);
    return bits ^ ((bits >> 31) & 0x7FFFFFFF);
}
inline float key_float(int32_t key) { return std::bit_cast<float>(key ^ ((key >> 31) & 0x7FFFFFFF)); }

// Exact rank-th smallest (0-based) key by bisection over the key range: each step is one SIMD
// counting pass, ≤ 32 steps (much cheaper than nth_element for the few-thousand-key inputs here).
inline int32_t kth_key(const int32_t* keys, size_t count, size_t rank) {
    const size_t whole = count / kLanes * kLanes;
    auto load = [keys](size_t i) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(keys + i)); };
    auto hsum = [](__m256i v) {
        __m128i x = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0x4E));
        return _mm_cvtsi128_si32(_mm_add_epi32(x, _mm_shuffle_epi32(x, 0xB1)));
    };
    int32_t lo = keys[0], hi = keys[0];
    if (whole > 0) {
        __m256i vmin = load(0), vmax = vmin;
        for (size_t i = kLanes; i < whole; i += kLanes) vmin = _mm256_min_epi32(vmin, load(i)), vmax = _mm256_max_epi32(vmax, load(i));
        alignas(32) int32_t lanes_min[kLanes], lanes_max[kLanes];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes_min), vmin);
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes_max), vmax);
        lo = *std::min_element(lanes_min, lanes_min + kLanes), hi = *std::max_element(lanes_max, lanes_max + kLanes);
    }
    for (size_t i = whole; i < count; ++i) lo = std::min(lo, keys[i]), hi = std::max(hi, keys[i]);
    // count of keys ≤ limit; the compare masks (-1 per hit) are accumulated in two vector registers
    auto count_le = [&](int32_t limit) {
        const __m256i bound = _mm256_set1_epi32(limit);
        __m256i acc0 = _mm256_setzero_si256(), acc1 = acc0;
        size_t i = 0;
        for (; i + 2 * kLanes <= whole; i += 2 * kLanes) {
            acc0 = _mm256_add_epi32(acc0, _mm256_cmpgt_epi32(load(i), bound));
            acc1 = _mm256_add_epi32(acc1, _mm256_cmpgt_epi32(load(i + kLanes), bound));
        }
        for (; i < whole; i += kLanes) acc0 = _mm256_add_epi32(acc0, _mm256_cmpgt_epi32(load(i), bound));
        size_t greater = static_cast<size_t>(-hsum(_mm256_add_epi32(acc0, acc1)));
        for (; i < count; ++i) greater += keys[i] > limit;
        return count - greater;
    };
    while (lo < hi) {
        const auto mid = static_cast<int32_t>(lo + ((static_cast<int64_t>(hi) - lo) >> 1));
        if (count_le(mid) > rank)
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}
}  // namespace select_detail

// Writes the ids of the n smallest scores into out. ids == nullptr means id = position.
inline void select_smallest(const float* scores, const uint32_t* ids, size_t count, size_t n,
                            std::vector<uint32_t>& out, SelectScratch& s) {
    using namespace select_detail;
    out.clear();
    auto id_at = [ids](size_t i) { return ids ? ids[i] : static_cast<uint32_t>(i); };
    if (n >= count) {
        for (size_t i = 0; i < count; ++i) out.push_back(id_at(i));
        return;
    }
    if (n == 0) return;

    // exact n smallest among m candidates: the n-th key v, then every key < v plus ties == v as needed
    auto pick_from = [&](const uint32_t* pos, size_t m) {
        auto position = [pos](size_t j) { return pos ? pos[j] : static_cast<uint32_t>(j); };
        s.keys.resize(m);
        for (size_t j = 0; j < m; ++j) s.keys[j] = float_key(scores[position(j)]);
        const int32_t v = kth_key(s.keys.data(), m, n - 1);
        for (size_t j = 0; j < m; ++j)
            if (s.keys[j] < v) out.push_back(id_at(position(j)));
        for (size_t j = 0; j < m && out.size() < n; ++j)
            if (s.keys[j] == v) out.push_back(id_at(position(j)));
    };

    if (count >= kMinSampled) {
        const size_t m = count / kStride;
        const size_t rank = std::min(m - 1, static_cast<size_t>(kSlack * n * m / count) + kSlackAbs);
        s.sample.resize(m);
        for (size_t j = 0; j < m; ++j) s.sample[j] = float_key(scores[j * kStride]);
        const float threshold = key_float(kth_key(s.sample.data(), m, rank));
        s.positions.resize(count + kLanes);
        const size_t kept = positions_at_most(scores, count, threshold, s.positions.data());
        if (kept >= n) return pick_from(s.positions.data(), kept);
    }
    pick_from(nullptr, count);
}

// Approximately the n smallest: every score ≤ the sampled rank-n quantile (≈ n ids, unordered).
// For selections whose size is only a tuning knob (fine cells); falls back to exact for small counts.
inline void select_about(const float* scores, const uint32_t* ids, size_t count, size_t n, std::vector<uint32_t>& out,
                         SelectScratch& s) {
    using namespace select_detail;
    if (n == 0 || n >= count || count < kMinSampled) return select_smallest(scores, ids, count, n, out, s);
    const size_t m = count / kStride;
    const size_t rank = std::min(m - 1, n * m / count);
    s.sample.resize(m);
    for (size_t j = 0; j < m; ++j) s.sample[j] = float_key(scores[j * kStride]);
    const float threshold = key_float(kth_key(s.sample.data(), m, rank));
    out.resize(count + kLanes);
    const size_t kept = positions_at_most(scores, count, threshold, out.data());
    out.resize(kept);
    if (ids)
        for (auto& v : out) v = ids[v];
}

}  // namespace hh
