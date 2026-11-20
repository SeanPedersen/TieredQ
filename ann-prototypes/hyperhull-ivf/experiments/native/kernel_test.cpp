// Checks SIMD kernels against scalar references; prints max errors and a micro-benchmark.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "kernels.hpp"
int main() {
    std::mt19937 rng(1); std::uniform_real_distribution<float> u(-1, 1); std::uniform_int_distribution<int> c4(0, 15), w8(-127, 127);
    const size_t dim = 1536, pdim = 128, n = 20000;
    std::vector<float> a(dim), b(dim); for (auto& x : a) x = u(rng); for (auto& x : b) x = u(rng);
    double l2 = 0, dt = 0; for (size_t i = 0; i < dim; ++i) { l2 += (a[i]-b[i])*(a[i]-b[i]); dt += a[i]*b[i]; }
    std::printf("%s l2 err %.2e dot err %.2e\n", hh::kSimdLevel, std::abs(hh::l2sq(a.data(), b.data(), dim) - l2) / l2, std::abs(hh::dot(a.data(), b.data(), dim) - dt) / std::abs(dt));
    std::vector<uint8_t> codes(n * pdim / 2); std::vector<int8_t> w(pdim); std::vector<int> raw(pdim);
    for (auto& x : w) x = w8(rng);
    long max_err = 0;
    for (size_t r = 0; r < n; ++r) {
        for (auto& x : raw) x = c4(rng);
        long ref = 0; for (size_t d = 0; d < pdim; ++d) ref += raw[d] * w[d];
        uint8_t* p = codes.data() + r * pdim / 2;
        for (size_t blk = 0; blk < pdim / 64; ++blk) for (size_t j = 0; j < 32; ++j) p[32*blk + j] = raw[64*blk + j] | (raw[64*blk + 32 + j] << 4);
        max_err = std::max(max_err, std::abs(ref - hh::dot_u4_s8(p, w.data(), pdim / 64)));
    }
    auto t = std::chrono::steady_clock::now(); long long sink = 0;
    for (int rep = 0; rep < 200; ++rep) for (size_t r = 0; r < n; ++r) sink += hh::dot_u4_s8(codes.data() + r * pdim / 2, w.data(), pdim / 64);
    double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t).count() / (200.0 * n);
    std::vector<float> vecs(1000 * dim); for (auto& x : vecs) x = u(rng); float fs = 0;
    t = std::chrono::steady_clock::now();
    for (int rep = 0; rep < 200; ++rep) for (size_t r = 0; r < 1000; ++r) fs += hh::l2sq(a.data(), vecs.data() + r * dim, dim);
    double ns_l2 = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t).count() / (200.0 * 1000);
    std::printf("%s 4-bit dot max abs err %ld | %.2f ns per 128-dim code | %.1f ns per 1536-dim L2 (sink %lld %.0f)\n", hh::kSimdLevel, max_err, ns, ns_l2, sink, fs);
}
