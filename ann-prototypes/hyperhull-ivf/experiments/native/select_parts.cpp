// Times the parts of hh::select_smallest (sample rank, SIMD threshold pass, exact pick) separately.
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "../../hyperhull-ivf/native/cpp/select.hpp"

using namespace hh::select_detail;
using Clock = std::chrono::steady_clock;

int main() {
    const size_t count = 22400, n = 800, m = count / kStride;
    std::mt19937 rng(1);
    std::normal_distribution<float> normal;
    std::vector<float> scores(count);
    for (auto& v : scores) v = normal(rng);
    std::vector<int32_t> sample(m), keys;
    std::vector<uint32_t> positions(count + kLanes);
    double t_sample = 0, t_pass = 0, t_pick = 0;
    size_t kept = 0;
    constexpr int kReps = 2000;
    for (int r = 0; r < kReps; ++r) {
        auto t0 = Clock::now();
        const size_t rank = std::min(m - 1, static_cast<size_t>(kSlack * n * m / count) + kSlackAbs);
        for (size_t j = 0; j < m; ++j) sample[j] = hh::select_detail::float_key(scores[j * kStride]);
        const float threshold = key_float(kth_key(sample.data(), m, rank));
        auto t1 = Clock::now();
        kept = positions_at_most(scores.data(), count, threshold, positions.data());
        auto t2 = Clock::now();
        keys.resize(kept);
        for (size_t j = 0; j < kept; ++j) keys[j] = float_key(scores[positions[j]]);
        volatile int32_t v = kth_key(keys.data(), kept, n - 1);
        (void)v;
        auto t3 = Clock::now();
        t_sample += std::chrono::duration<double, std::micro>(t1 - t0).count();
        t_pass += std::chrono::duration<double, std::micro>(t2 - t1).count();
        t_pick += std::chrono::duration<double, std::micro>(t3 - t2).count();
    }
    std::printf("sample %.2f us, pass %.2f us, pick %.2f us, kept %zu\n", t_sample / kReps, t_pass / kReps, t_pick / kReps, kept);
}
