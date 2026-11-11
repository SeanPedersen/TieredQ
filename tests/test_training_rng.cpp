// Check training randomness contracts independently of codec arithmetic.
#include "training_rng.h"
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>

void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

int main() {
  try {
    using tq::tieredq_detail::TrainingRng;
    constexpr uint64_t seed = 42;
    constexpr size_t population = 100, selected = 17, samples = 65536;
    TrainingRng first(seed), second(seed);
    auto indices = first.choice(population, selected);
    require(indices == second.choice(population, selected), "sampling seed reproducibility");
    require(indices.size() == selected, "sample count");
    require(std::set<size_t>(indices.begin(), indices.end()).size() == selected, "sampling without replacement");
    for (auto index : indices) require(index < population, "sample bounds");
    auto all = first.choice(population, population);
    require(std::set<size_t>(all.begin(), all.end()).size() == population, "whole-population sample");
    require(first.choice(0, 0).empty(), "empty sample");
    bool rejected = false;
    try { first.choice(0, 1); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid sample rejected");
    TrainingRng gaussian(seed), repeated(seed);
    double sum = 0, squared_sum = 0;
    for (size_t i = 0; i < samples; ++i) {
      const double value = gaussian.normal();
      require(std::isfinite(value) && value == repeated.normal(), "normal seed reproducibility");
      sum += value;
      squared_sum += value * value;
    }
    constexpr double mean_tolerance = .02, variance_tolerance = .03;
    const double mean = sum / samples;
    const double variance = squared_sum / samples - mean * mean;
    require(std::abs(mean) < mean_tolerance, "normal mean");
    require(std::abs(variance - 1) < variance_tolerance, "normal variance");
    std::cout << "PASS training RNG\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
