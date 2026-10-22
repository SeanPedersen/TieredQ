// NumPy-compatible SeedSequence -> PCG64, bounded integers and double normals.
// Algorithms adapted from NumPy v2.3.3 random sources; see detail/LICENSE.numpy.
// No Python or NumPy runtime dependency. Seeds are unsigned 64-bit integers.
#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include "detail/numpy_ziggurat.h"

namespace dv::tiered {
class NumpyRng {
 public:
  explicit NumpyRng(uint64_t seed) {
    uint32_t pool[4], hc = 0x43b0d7e5;
    const uint32_t entropy[] = {(uint32_t)seed, (uint32_t)(seed >> 32), 0, 0};
    for (int i = 0; i < 4; i++) pool[i] = hashmix(entropy[i], hc);
    for (int src = 0; src < 4; src++) for (int dst = 0; dst < 4; dst++) if (src != dst) {
      uint32_t v = 0xca01f9ddU * pool[dst] - 0x4973f715U * hashmix(pool[src], hc);
      pool[dst] = v ^ (v >> 16);
    }
    uint32_t words[8]; hc = 0x8b51f9dd;
    for (int i = 0; i < 8; i++) {
      uint32_t v = pool[i % 4] ^ hc; hc *= 0x58f38ded; v *= hc; words[i] = v ^ (v >> 16);
    }
    uint64_t v[4]; for (int i = 0; i < 4; i++) v[i] = (uint64_t)words[2 * i] | ((uint64_t)words[2 * i + 1] << 32);
    state_ = 0; inc_ = ((((U128)v[2] << 64) | v[3]) << 1) | 1;
    step(); state_ += ((U128)v[0] << 64) | v[1]; step();
  }
  uint64_t next64() {
    step(); const uint64_t x = (uint64_t)(state_ >> 64) ^ (uint64_t)state_;
    const unsigned r = state_ >> 122; return (x >> r) | (x << ((-r) & 63));
  }
  uint32_t next32() {
    if (cached_) { cached_ = false; return cache_; }
    const uint64_t v = next64(); cached_ = true; cache_ = v >> 32; return (uint32_t)v;
  }
  uint64_t bounded(uint64_t inclusive) {
    if (!inclusive) return 0;
    if (inclusive <= UINT32_MAX) {
      if (inclusive == UINT32_MAX) return next32();
      const uint32_t range = inclusive + 1, threshold = (UINT32_MAX - (uint32_t)inclusive) % range;
      for (;;) { const uint64_t m = (uint64_t)next32() * range; if ((uint32_t)m >= threshold) return m >> 32; }
    }
    if (inclusive == UINT64_MAX) return next64();
    const uint64_t range = inclusive + 1, threshold = (UINT64_MAX - inclusive) % range;
    for (;;) { const U128 m = (U128)next64() * range; if ((uint64_t)m >= threshold) return m >> 64; }
  }
  double uniform() { return (next64() >> 11) * 0x1.0p-53; }
  double normal() {
    using namespace detail;
    for (;;) {
      uint64_t r = next64(); const int idx = r & 255; r >>= 8;
      const int sign = r & 1; const uint64_t abs = (r >> 1) & 0x000fffffffffffffULL;
      double x = abs * wi_double[idx]; if (sign) x = -x;
      if (abs < ki_double[idx]) return x;
      if (idx == 0) {
        for (;;) {
          const double xx = -ziggurat_nor_inv_r * std::log1p(-uniform()), yy = -std::log1p(-uniform());
          if (yy + yy > xx * xx) return ((abs >> 8) & 1) ? -(ziggurat_nor_r + xx) : ziggurat_nor_r + xx;
        }
      } else if ((fi_double[idx - 1] - fi_double[idx]) * uniform() + fi_double[idx] < std::exp(-0.5 * x * x)) return x;
    }
  }
  std::vector<int64_t> choice(size_t n, size_t count) {
    if (count > n) throw std::invalid_argument("choice size exceeds population");
    if (n > 10000 && count > n / 50) {
      std::vector<int64_t> all(n); std::iota(all.begin(), all.end(), 0);
      shuffle(all, std::max<size_t>(n - count, 1)); return {all.end() - count, all.end()};
    }
    std::unordered_set<uint64_t> seen; std::vector<int64_t> out; out.reserve(count);
    for (size_t j = n - count; j < n; j++) {
      const uint64_t v = bounded(j), chosen = seen.count(v) ? j : v;
      seen.insert(chosen); out.push_back(chosen);
    }
    shuffle(out, 1); return out;
  }
 private:
  using U128 = unsigned __int128;
  U128 state_, inc_; bool cached_ = false; uint32_t cache_ = 0;
  static uint32_t hashmix(uint32_t v, uint32_t& hc) { v ^= hc; hc *= 0x931e8875; v *= hc; return v ^ (v >> 16); }
  void step() { state_ = state_ * (((U128)2549297995355413924ULL << 64) | 4865540595714422341ULL) + inc_; }
  void shuffle(std::vector<int64_t>& v, size_t first) {
    for (size_t i = v.size(); i > first; ) { --i; std::swap(v[i], v[bounded(i)]); }
  }
};
} // namespace dv::tiered
