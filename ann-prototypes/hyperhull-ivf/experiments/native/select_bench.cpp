// Microbenchmark of hh::select_smallest: count scores → n smallest (random normal scores).
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "../../hyperhull-ivf/native/cpp/select.hpp"

int main() {
    std::mt19937 rng(1);
    std::normal_distribution<float> normal;
    for (auto [count, n] : {std::pair{22400ul, 800ul}, {22400ul, 100ul}, {4400ul, 300ul}, {8192ul, 1536ul}}) {
        std::vector<float> scores(count);
        for (auto& v : scores) v = normal(rng);
        hh::SelectScratch s;
        std::vector<uint32_t> out;
        constexpr int kReps = 2000;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < kReps; ++r) {
            scores[r % count] += 1e-3f;
            hh::select_smallest(scores.data(), nullptr, count, n, out, s);
        }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / kReps;
        std::printf("count %6zu n %5zu: %6.1f us (out %zu)\n", count, n, us, out.size());
    }
}
