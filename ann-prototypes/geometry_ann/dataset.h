// Benchmark-only datasets and exact ground truth, independent of index implementation.
#pragma once
#include "index.h"
#include <string>
namespace geometry_ann {
struct Dataset {
    std::size_t dimension = 0;
    std::vector<float> base, queries;
    std::vector<std::vector<Id>> truth;
};
Dataset synthetic(std::size_t count, std::size_t queries, std::size_t dimension, std::uint32_t seed);
Dataset load_dataset(const std::string& base, const std::string& queries, const std::string& truth,
                     std::size_t query_limit);
std::vector<Neighbor> exact(std::span<const float> base, std::span<const float> query, std::size_t k);
}
