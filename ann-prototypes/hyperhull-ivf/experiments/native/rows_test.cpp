#include <cstdio>
#include <random>
#include <vector>
#include "kernels.hpp"
int main() {
    std::mt19937 rng(5); std::vector<uint8_t> codes(37 * 64); std::vector<int8_t> w(128); std::vector<int32_t> out(37);
    for (auto& c : codes) c = rng(); for (auto& x : w) x = int8_t(int(rng() % 255) - 127);
    hh::dot_u4_s8_rows(codes.data(), 37, 64, w.data(), 2, out.data()); int bad = 0;
    for (int r = 0; r < 37; ++r) bad += out[r] != hh::dot_u4_s8(codes.data() + r * 64, w.data(), 2);
    std::printf("%s dot_u4_s8_rows mismatches %d\n", hh::kSimdLevel, bad); return bad;
}
