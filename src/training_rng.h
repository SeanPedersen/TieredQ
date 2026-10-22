// Seeded training samples and Gaussian rotations using the C++ standard library.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace dv::tieredq_detail {
class TrainingRng {
 public:
  explicit TrainingRng(uint64_t seed) : engine_(seed) {}

  double normal() { return normal_(engine_); }

  std::vector<size_t> choice(size_t population, size_t count) {
    if (count > population) throw std::invalid_argument("sample exceeds population");
    if (count == 0) return {};
    std::vector<size_t> indices(population);
    std::iota(indices.begin(), indices.end(), size_t{0});
    for (size_t first = 0; first < count; ++first) {
      std::uniform_int_distribution<size_t> select(first, population - 1);
      std::swap(indices[first], indices[select(engine_)]);
    }
    indices.resize(count);
    return indices;
  }

 private:
  std::mt19937_64 engine_;
  std::normal_distribution<double> normal_;
};
}  // namespace dv::tieredq_detail
