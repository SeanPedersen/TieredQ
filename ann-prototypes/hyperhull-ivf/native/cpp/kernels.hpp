// SIMD distance kernels: AVX2 + FMA; 512-bit variants only with -DHH_AVX512 (on Skylake-X they drop the
// core to its lowest AVX-512 turbo licence, which costs more than they gain in this memory-bound search).
//
// 4-bit code layout: dims are grouped in blocks of 64; within block b, byte j holds
// dim 64b+j in the low nibble and dim 64b+32+j in the high nibble (32 bytes per block).
// Query weights are int8 in natural dim order, so one block = two maddubs over 32 lanes.
#pragma once

#if defined(HH_AVX512) && defined(__AVX512BW__)
#define HH_KERNELS_512 1
#endif

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace hh {

inline float hsum(__m256 v) {
    __m128 x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    x = _mm_hadd_ps(x, x);
    x = _mm_hadd_ps(x, x);
    return _mm_cvtss_f32(x);
}

inline int32_t hsum(__m256i v) {
    __m128i x = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    x = _mm_hadd_epi32(x, x);
    x = _mm_hadd_epi32(x, x);
    return _mm_cvtsi128_si32(x);
}

#if defined(HH_KERNELS_512)
inline float l2sq(const float* a, const float* b, size_t dim) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        const __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        const __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16));
        s0 = _mm512_fmadd_ps(d0, d0, s0);
        s1 = _mm512_fmadd_ps(d1, d1, s1);
    }
    if (i + 16 <= dim) {
        const __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        s0 = _mm512_fmadd_ps(d0, d0, s0);
        i += 16;
    }
    float s = _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
    for (; i < dim; ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
    return s;
}

inline float dot(const float* a, const float* b, size_t dim) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s0);
        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), s1);
    }
    if (i + 16 <= dim) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s0);
        i += 16;
    }
    float s = _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
    for (; i < dim; ++i) s += a[i] * b[i];
    return s;
}
#else
inline float l2sq(const float* a, const float* b, size_t dim) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        s1 = _mm256_fmadd_ps(d1, d1, s1);
    }
    float s = hsum(_mm256_add_ps(s0, s1));
    for (; i < dim; ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
    return s;
}

inline float dot(const float* a, const float* b, size_t dim) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    float s = hsum(_mm256_add_ps(s0, s1));
    for (; i < dim; ++i) s += a[i] * b[i];
    return s;
}

#endif

// Σ_d code_d · w_d for packed 4-bit codes (n_blocks × 64 dims) and int8 weights.
#if defined(HH_KERNELS_512)
inline int32_t dot_u4_s8(const uint8_t* codes, const int8_t* w, size_t n_blocks) {
    const __m512i low_mask = _mm512_set1_epi8(0x0F);
    const __m512i ones = _mm512_set1_epi16(1);
    __m512i acc = _mm512_setzero_si512();
    for (size_t b = 0; b < n_blocks; ++b) {
        // lanes 0..31 = low nibbles (dims 64b..64b+31), lanes 32..63 = high nibbles → natural dim order
        const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + 32 * b));
        const __m512i both = _mm512_inserti64x4(_mm512_castsi256_si512(packed), _mm256_srli_epi16(packed, 4), 1);
        const __m512i nibbles = _mm512_and_si512(both, low_mask);
        const __m512i weights = _mm512_loadu_si512(w + 64 * b);
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(_mm512_maddubs_epi16(nibbles, weights), ones));
    }
    return _mm512_reduce_add_epi32(acc);
}
#else
inline int32_t dot_u4_s8(const uint8_t* codes, const int8_t* w, size_t n_blocks) {
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    for (size_t b = 0; b < n_blocks; ++b) {
        const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + 32 * b));
        const __m256i lo = _mm256_and_si256(packed, low_mask);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
        const __m256i w_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 64 * b));
        const __m256i w_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 64 * b + 32));
        // |pair sum| <= 2·15·127 per maddubs → sum of both fits int16
        const __m256i pairs = _mm256_add_epi16(_mm256_maddubs_epi16(lo, w_lo), _mm256_maddubs_epi16(hi, w_hi));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, ones));
    }
    return hsum(acc);
}
#endif

// Coarse centroids are stored in blocks of kCentroidBlock, dim-major inside a block (block[d·16 + j]).
constexpr size_t kCentroidBlock = 16;

// Projection GEMM micro-kernel: out[j·out_stride + o] += Σ_d q[j·q_stride + d] · strip[d·16 + o] for
// Q ≤ kTileQueries queries and the 16 outputs of one strip of the transposed basis. Query values are
// broadcast, so the 2·Q accumulators need no horizontal reduction. With Q = 6 a step is 8 loads per
// 12 FMAs (FMA-bound); callers keep the strip chunk L1-resident (K-blocking).
constexpr size_t kStripOutputs = 16;
constexpr size_t kTileQueries = 6;
template <size_t Q>
inline void project_strip(const float* strip, size_t dim, const float* q, size_t q_stride, float* out, size_t out_stride) {
    __m256 acc[Q][2];
#pragma GCC unroll 8
    for (size_t j = 0; j < Q; ++j) {
        acc[j][0] = _mm256_loadu_ps(out + j * out_stride);
        acc[j][1] = _mm256_loadu_ps(out + j * out_stride + 8);
    }
    for (size_t d = 0; d < dim; ++d) {
        const __m256 b0 = _mm256_loadu_ps(strip + d * kStripOutputs), b1 = _mm256_loadu_ps(strip + d * kStripOutputs + 8);
#pragma GCC unroll 8
        for (size_t j = 0; j < Q; ++j) {
            const __m256 a = _mm256_broadcast_ss(q + j * q_stride + d);
            acc[j][0] = _mm256_fmadd_ps(a, b0, acc[j][0]);
            acc[j][1] = _mm256_fmadd_ps(a, b1, acc[j][1]);
        }
    }
#pragma GCC unroll 8
    for (size_t j = 0; j < Q; ++j) {
        _mm256_storeu_ps(out + j * out_stride, acc[j][0]);
        _mm256_storeu_ps(out + j * out_stride + 8, acc[j][1]);
    }
}

// project_strip over `count` queries: full kTileQueries tiles, then one smaller tile for the rest.
inline void project_strip_rows(const float* strip, size_t dim, const float* q, size_t q_stride, size_t count, float* out,
                               size_t out_stride) {
    size_t j = 0;
    for (; j + kTileQueries <= count; j += kTileQueries)
        project_strip<kTileQueries>(strip, dim, q + j * q_stride, q_stride, out + j * out_stride, out_stride);
    q += j * q_stride, out += j * out_stride;
    switch (count - j) {
        case 5: return project_strip<5>(strip, dim, q, q_stride, out, out_stride);
        case 4: return project_strip<4>(strip, dim, q, q_stride, out, out_stride);
        case 3: return project_strip<3>(strip, dim, q, q_stride, out, out_stride);
        case 2: return project_strip<2>(strip, dim, q, q_stride, out, out_stride);
        case 1: return project_strip<1>(strip, dim, q, q_stride, out, out_stride);
        default: return;
    }
}

// Dot product of an fp16 row with a float vector (dim % 16 == 0).
inline float dot_f16(const uint16_t* a, const float* b, size_t dim) {
#if defined(HH_KERNELS_512)
    __m512 s0 = _mm512_setzero_ps();
    for (size_t i = 0; i < dim; i += 16) {
        const __m512 x = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i)));
        s0 = _mm512_fmadd_ps(x, _mm512_loadu_ps(b + i), s0);
    }
    return _mm512_reduce_add_ps(s0);
#else
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    for (size_t i = 0; i < dim; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i))), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 8))), _mm256_loadu_ps(b + i + 8), s1);
    }
    return hsum(_mm256_add_ps(s0, s1));
#endif
}

inline void f16_to_f32(const uint16_t* src, size_t n, float* dst) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i))));
    for (; i < n; ++i) dst[i] = _cvtsh_ss(src[i]);
}

inline void to_f16(const float* src, size_t n, uint16_t* dst) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
    for (; i < n; ++i) dst[i] = _cvtss_sh(src[i], _MM_FROUND_TO_NEAREST_INT);
}

constexpr size_t kCacheLine = 64;
inline void prefetch(const void* p, size_t bytes) {
    const char* c = static_cast<const char*>(p);
    for (size_t off = 0; off < bytes; off += kCacheLine) _mm_prefetch(c + off, _MM_HINT_T0);
}

// dot_u4_s8 for `count` code rows spaced row_bytes apart (≥ n_blocks·32), 4 rows per pass with a
// shared horizontal reduction. out[r] = Σ_d code_rd · w_d.
inline __m128i reduce4(__m256i a0, __m256i a1, __m256i a2, __m256i a3) {
    const __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(a0, a1), _mm256_hadd_epi32(a2, a3));
    return _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
}

inline void dot_u4_s8_rows(const uint8_t* codes, size_t count, size_t row_bytes, const int8_t* w, size_t n_blocks,
                           int32_t* out) {
    size_t r = 0;
#if defined(HH_KERNELS_512)
    const __m512i low_mask = _mm512_set1_epi8(0x0F);
    const __m512i ones = _mm512_set1_epi16(1);
    auto row_block = [&](const uint8_t* row, size_t b) {
        const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + 32 * b));
        const __m512i nibbles = _mm512_and_si512(
            _mm512_inserti64x4(_mm512_castsi256_si512(packed), _mm256_srli_epi16(packed, 4), 1), low_mask);
        return _mm512_madd_epi16(_mm512_maddubs_epi16(nibbles, _mm512_loadu_si512(w + 64 * b)), ones);
    };
    auto halve = [](__m512i v) { return _mm256_add_epi32(_mm512_castsi512_si256(v), _mm512_extracti64x4_epi64(v, 1)); };
    for (; r + 4 <= count; r += 4) {
        const uint8_t* row = codes + r * row_bytes;
        __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
        for (size_t b = 0; b < n_blocks; ++b) {
            a0 = _mm512_add_epi32(a0, row_block(row, b));
            a1 = _mm512_add_epi32(a1, row_block(row + row_bytes, b));
            a2 = _mm512_add_epi32(a2, row_block(row + 2 * row_bytes, b));
            a3 = _mm512_add_epi32(a3, row_block(row + 3 * row_bytes, b));
        }
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + r), reduce4(halve(a0), halve(a1), halve(a2), halve(a3)));
    }
#else
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    auto row_block = [&](const uint8_t* row, size_t b) {
        const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + 32 * b));
        const __m256i lo = _mm256_and_si256(packed, low_mask);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
        const __m256i w_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 64 * b));
        const __m256i w_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 64 * b + 32));
        return _mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(lo, w_lo), _mm256_maddubs_epi16(hi, w_hi)), ones);
    };
    for (; r + 4 <= count; r += 4) {
        const uint8_t* row = codes + r * row_bytes;
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
        for (size_t b = 0; b < n_blocks; ++b) {
            a0 = _mm256_add_epi32(a0, row_block(row, b));
            a1 = _mm256_add_epi32(a1, row_block(row + row_bytes, b));
            a2 = _mm256_add_epi32(a2, row_block(row + 2 * row_bytes, b));
            a3 = _mm256_add_epi32(a3, row_block(row + 3 * row_bytes, b));
        }
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + r), reduce4(a0, a1, a2, a3));
    }
#endif
    for (; r < count; ++r) out[r] = dot_u4_s8(codes + r * row_bytes, w, n_blocks);
}

// dot_u4_s8 of 4 code rows at arbitrary addresses (AVX2): [Σ_d code_rd · w_d for r = 0..3].
inline __m128i dot_u4_s8_4rows(const uint8_t* r0, const uint8_t* r1, const uint8_t* r2, const uint8_t* r3, const int8_t* w,
                               size_t n_blocks) {
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    for (size_t b = 0; b < n_blocks; ++b) {
        const __m256i w_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 64 * b));
        const __m256i w_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 64 * b + 32));
        auto row_block = [&](const uint8_t* row) {
            const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + 32 * b));
            const __m256i lo = _mm256_and_si256(packed, low_mask);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
            return _mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(lo, w_lo), _mm256_maddubs_epi16(hi, w_hi)), ones);
        };
        a0 = _mm256_add_epi32(a0, row_block(r0));
        a1 = _mm256_add_epi32(a1, row_block(r1));
        a2 = _mm256_add_epi32(a2, row_block(r2));
        a3 = _mm256_add_epi32(a3, row_block(r3));
    }
    return reduce4(a0, a1, a2, a3);
}

// Query weights of NB 64-dim blocks held in registers (the int8 weights may alias any store, so a
// compiler would otherwise reload them for every row group).
template <size_t NB>
struct U4Weights {
    __m256i lo[NB], hi[NB];
    explicit U4Weights(const int8_t* w) {
        for (size_t b = 0; b < NB; ++b) {
            lo[b] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 64 * b));
            hi[b] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 64 * b + 32));
        }
    }
    __m128i operator()(const uint8_t* r0, const uint8_t* r1, const uint8_t* r2, const uint8_t* r3) const {
        const __m256i low_mask = _mm256_set1_epi8(0x0F);
        const __m256i ones = _mm256_set1_epi16(1);
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
#pragma GCC unroll 8
        for (size_t b = 0; b < NB; ++b) {
            auto row_block = [&](const uint8_t* row) {
                const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + 32 * b));
                const __m256i nib_lo = _mm256_and_si256(packed, low_mask);
                const __m256i nib_hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_mask);
                return _mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(nib_lo, lo[b]), _mm256_maddubs_epi16(nib_hi, hi[b])), ones);
            };
            a0 = _mm256_add_epi32(a0, row_block(r0));
            a1 = _mm256_add_epi32(a1, row_block(r1));
            a2 = _mm256_add_epi32(a2, row_block(r2));
            a3 = _mm256_add_epi32(a3, row_block(r3));
        }
        return reduce4(a0, a1, a2, a3);
    }
};

// 2-bit codes: with Q = dims/4, byte j holds dims j, Q+j, 2Q+j, 3Q+j in bits 0-1, 2-3, 4-5, 6-7
// (Q must be a multiple of 32). out[r] = Σ_d code_rd · w_d for `count` rows spaced row_bytes apart.
inline __m256i u2_chunk(const uint8_t* row, size_t c, size_t quarter, const int8_t* w) {
    const __m256i mask = _mm256_set1_epi8(0x03);
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + c));
    __m256i pairs = _mm256_setzero_si256();  // |sum| <= 4 · 2·3·127, fits int16
    for (int k = 0; k < 4; ++k) {
        const __m256i codes = _mm256_and_si256(_mm256_srli_epi16(x, 2 * k), mask);
        const __m256i wk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + k * quarter + c));
        pairs = _mm256_add_epi16(pairs, _mm256_maddubs_epi16(codes, wk));
    }
    return _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
}

inline void dot_u2_s8_rows(const uint8_t* codes, size_t count, size_t row_bytes, const int8_t* w, size_t dims,
                           int32_t* out) {
    const size_t quarter = dims / 4;
    size_t r = 0;
    for (; r + 4 <= count; r += 4) {
        const uint8_t* row = codes + r * row_bytes;
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
        for (size_t c = 0; c < quarter; c += 32) {
            a0 = _mm256_add_epi32(a0, u2_chunk(row, c, quarter, w));
            a1 = _mm256_add_epi32(a1, u2_chunk(row + row_bytes, c, quarter, w));
            a2 = _mm256_add_epi32(a2, u2_chunk(row + 2 * row_bytes, c, quarter, w));
            a3 = _mm256_add_epi32(a3, u2_chunk(row + 3 * row_bytes, c, quarter, w));
        }
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + r), reduce4(a0, a1, a2, a3));
    }
    for (; r < count; ++r) {
        __m256i acc = _mm256_setzero_si256();
        for (size_t c = 0; c < quarter; c += 32) acc = _mm256_add_epi32(acc, u2_chunk(codes + r * row_bytes, c, quarter, w));
        out[r] = hsum(acc);
    }
}

// dot_u2_s8_rows for 4 code rows at arbitrary addresses.
inline __m128i dot_u2_s8_4rows(const uint8_t* r0, const uint8_t* r1, const uint8_t* r2, const uint8_t* r3, const int8_t* w,
                               size_t dims) {
    const size_t quarter = dims / 4;
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    for (size_t c = 0; c < quarter; c += 32) {
        a0 = _mm256_add_epi32(a0, u2_chunk(r0, c, quarter, w));
        a1 = _mm256_add_epi32(a1, u2_chunk(r1, c, quarter, w));
        a2 = _mm256_add_epi32(a2, u2_chunk(r2, c, quarter, w));
        a3 = _mm256_add_epi32(a3, u2_chunk(r3, c, quarter, w));
    }
    return reduce4(a0, a1, a2, a3);
}

// Which kernel set this build uses (reported by the Python module).
constexpr const char* kSimdLevel =
#if defined(HH_KERNELS_512)
    "avx512";
#else
    "avx2";
#endif

}  // namespace hh
