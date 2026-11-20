// select_smallest vs sort on random, skewed and tied inputs.
#include <algorithm>
#include <cstdio>
#include <random>
#include "select.hpp"
int main() {
    std::mt19937 rng(3); hh::SelectScratch s; std::vector<uint32_t> out; int bad = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const size_t count = 1 + rng() % 5000, n = rng() % (count + 3);
        std::vector<float> v(count);
        const int kind = trial % 4;
        for (auto& x : v) {
            if (kind == 0) x = std::uniform_real_distribution<float>(-5, 5)(rng);
            if (kind == 1) x = std::exponential_distribution<float>(1)(rng) * (rng() % 50 == 0 ? 1e6f : 1);  // outliers
            if (kind == 2) x = float(rng() % 7);                                                              // ties
            if (kind == 3) x = 3.0f;                                                                          // constant
        }
        hh::select_smallest(v.data(), nullptr, count, n, out, s);
        std::vector<float> got; for (auto i : out) got.push_back(v[i]);
        std::vector<float> want = v; std::sort(want.begin(), want.end()); want.resize(std::min(n, count));
        std::sort(got.begin(), got.end());
        std::vector<uint32_t> uniq = out; std::sort(uniq.begin(), uniq.end());
        if (got != want || std::adjacent_find(uniq.begin(), uniq.end()) != uniq.end()) ++bad;
    }
    std::printf("select_smallest mismatches: %d / 3000\n", bad);
    return bad != 0;
}
