// Issue #24 microbenchmark: query projection onto int8 rank panels, d inputs x R ranks.
// k_float: current kernel (int8 -> fp32 convert + FMA), layout [block][input][panel][16].
// k_madd16: query as int16 (12-bit), panels int8 widened to int16, pmaddwd; layout [block][input pair][panel][16][2].
// k_maddubs: query as u8 (offset 64, 7-bit), panels int8, pmaddubsw + pmaddwd; layout [block][input quad][panel][16][4].
// k_batch: fp32 query batch B x panels (current layout), panel decoded once per B queries.
// Build: g++ -O3 -march=native -std=c++20 kernels.cpp -o kernels; run: ./kernels RANKS (pinned to one CPU).
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <cstring>
#include <vector>

constexpr size_t lanes = 16, block = 16;
using Clock = std::chrono::steady_clock;

template <size_t P>
void k_float(const int8_t* panels, size_t d, const float* x, float* out) {
  __m512 acc[P];
  for (auto& a : acc) a = _mm512_setzero_ps();
  for (size_t i = 0; i < d; ++i) {
    const __m512 v = _mm512_set1_ps(x[i]);
    for (size_t b = 0; b < P; ++b)
      acc[b] = _mm512_fmadd_ps(v, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(panels + (i * P + b) * lanes)))), acc[b]);
  }
  for (size_t b = 0; b < P; ++b) _mm512_storeu_ps(out + b * lanes, acc[b]);
}
// 32 bytes = 16 ranks x 2 inputs, widened to 16-bit pairs.
template <size_t P>
void k_madd16(const int8_t* panels, size_t d, const int16_t* x, float* out) {
  __m512i acc[P];
  for (auto& a : acc) a = _mm512_setzero_si512();
  for (size_t i = 0; i < d; i += 2) {
    int32_t pair; std::memcpy(&pair, x + i, 4);
    const __m512i v = _mm512_set1_epi32(pair);
    for (size_t b = 0; b < P; ++b)
      acc[b] = _mm512_add_epi32(acc[b], _mm512_madd_epi16(v, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(panels + (i / 2 * P + b) * 32)))));
  }
  for (size_t b = 0; b < P; ++b) _mm512_storeu_ps(out + b * lanes, _mm512_cvtepi32_ps(acc[b]));
}
template <size_t P>
void k_maddubs(const int8_t* panels, size_t d, const uint8_t* x, float* out) {
  __m512i acc[P];
  const __m512i ones = _mm512_set1_epi16(1);
  for (auto& a : acc) a = _mm512_setzero_si512();
  for (size_t i = 0; i < d; i += 4) {
    int32_t quad; std::memcpy(&quad, x + i, 4);
    const __m512i v = _mm512_set1_epi32(quad);
    for (size_t b = 0; b < P; ++b)
      acc[b] = _mm512_add_epi32(acc[b], _mm512_madd_epi16(_mm512_maddubs_epi16(v, _mm512_loadu_si512(panels + (i / 4 * P + b) * 64)), ones));
  }
  for (size_t b = 0; b < P; ++b) _mm512_storeu_ps(out + b * lanes, _mm512_cvtepi32_ps(acc[b]));
}
// B queries x P panels of accumulators; P * B <= 28 registers.
template <size_t P, size_t B>
void k_batch(const int8_t* panels, size_t d, const float* x, float* out, size_t ranks) {
  __m512 acc[B][P];
  for (auto& row : acc) for (auto& a : row) a = _mm512_setzero_ps();
  for (size_t i = 0; i < d; ++i) {
    __m512 p[P];
    for (size_t b = 0; b < P; ++b) p[b] = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(panels + (i * P + b) * lanes))));
    for (size_t q = 0; q < B; ++q) {
      const __m512 v = _mm512_set1_ps(x[q * d + i]);
      for (size_t b = 0; b < P; ++b) acc[q][b] = _mm512_fmadd_ps(v, p[b], acc[q][b]);
    }
  }
  for (size_t q = 0; q < B; ++q) for (size_t b = 0; b < P; ++b) _mm512_storeu_ps(out + q * ranks + b * lanes, acc[q][b]);
}

template <class F>
double time_us(F&& f, size_t rounds) {
  f();
  const auto start = Clock::now();
  for (size_t r = 0; r < rounds; ++r) f();
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count() / double(rounds);
}

int main(int argc, char** argv) {
  const size_t d = 1536, ranks = argc > 1 ? std::stoul(argv[1]) : 1056, rounds = 2000, groups = ranks / (lanes * block);
  std::mt19937 rng(0);
  std::normal_distribution<float> normal;
  std::vector<int8_t> panels(d * ranks);
  for (auto& p : panels) p = int8_t(std::clamp(std::lround(normal(rng) * 40), -127L, 127L));
  std::vector<float> x(d * 4), out(ranks * 4);
  for (auto& v : x) v = normal(rng);
  std::vector<int16_t> x16(d);
  std::vector<uint8_t> x8(d);
  for (size_t i = 0; i < d; ++i) { x16[i] = int16_t(std::lround(x[i] * 400)); x8[i] = uint8_t(64 + std::clamp(std::lround(x[i] * 16), -64L, 64L)); }
  const size_t stride = d * block * lanes;  // bytes per 16-panel group
  const double f = time_us([&] { for (size_t g = 0; g < groups; ++g) k_float<block>(panels.data() + g * stride, d, x.data(), out.data() + g * 256); }, rounds);
  const double m = time_us([&] { for (size_t g = 0; g < groups; ++g) k_madd16<block>(panels.data() + g * stride, d, x16.data(), out.data() + g * 256); }, rounds);
  const double u = time_us([&] { for (size_t g = 0; g < groups; ++g) k_maddubs<block>(panels.data() + g * stride, d, x8.data(), out.data() + g * 256); }, rounds);
  // Batch of 4 queries over groups of 6 panels (24 accumulators).
  constexpr size_t bp = 6, bq = 4;
  const size_t bgroups = ranks / (lanes * bp), bstride = d * bp * lanes;
  const double b = time_us([&] { for (size_t g = 0; g < bgroups; ++g) k_batch<bp, bq>(panels.data() + g * bstride, d, x.data(), out.data() + g * bp * lanes, ranks); }, rounds) / bq;
  std::printf("ranks %zu (%.2f MB): float %.1f us  madd16 %.1f us  maddubs %.1f us  batch4 %.1f us/query\n", ranks,
              double(panels.size()) / 1e6, f, m, u, b);
}
