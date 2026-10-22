// Standalone training, append, exhaustive search, and persistence benchmark.
#include <tieredq/tieredq.h>
#include <chrono>
#include <iostream>
#include <random>

int main() {
  constexpr size_t dimensions = 64, rows = 1024, neighbors = 10;
  std::mt19937 random(42);
  std::normal_distribution<float> normal;
  std::vector<float> data(rows * dimensions);
  for (auto& value : data) value = normal(random);
  const auto start = std::chrono::steady_clock::now();
  auto codec = dv::TieredQ::train({dimensions, dimensions * 2, 16}, data.data(), rows, 42);
  dv::TieredQIndex index(std::move(codec));
  index.add(data.data(), rows);
  const auto hits = index.search(data.data(), neighbors);
  std::cout << "rows=" << index.size() << " payload_bytes=" << index.payload_bytes()
            << " train_encode_search_ms="
            << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() << '\n';
  for (const auto& hit : hits) std::cout << hit.id << ' ' << hit.score << '\n';
  index.save("example.tq");
  return dv::TieredQIndex::load("example.tq").size() == rows ? 0 : 1;
}
