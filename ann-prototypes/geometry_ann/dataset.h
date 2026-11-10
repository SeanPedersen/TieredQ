// Benchmark-only datasets and exact ground truth, independent of index implementation.
#pragma once
#include "index.h"
#include <string>
#include <memory>
namespace geometry_ann {
struct Dataset {
    std::size_t dimension = 0, base_rows = 0;
    std::vector<float> base_storage, queries;
    std::shared_ptr<void> base_mapping;
    std::span<const float> base;
    std::size_t count() const { return base_rows; }
    void release_base() { base = {}; base_mapping.reset(); std::vector<float>().swap(base_storage); }
    Dataset() = default;
    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;
    Dataset(Dataset&&) = default;
    Dataset& operator=(Dataset&&) = default;
    std::vector<std::vector<Id>> truth;
};
struct QuantizedDataset {
    std::shared_ptr<void> mapping;
    std::vector<float> scales, offsets;
    std::span<const std::int8_t> codes;
    QuantizedInput view() const { return {codes,scales,offsets}; }
    void release_codes() { codes = {}; mapping.reset(); }
};
QuantizedDataset load_quantized(const std::string& codes, const std::string& scales, std::size_t count, std::size_t dimension, const std::string& offsets = "");
Dataset synthetic(std::size_t count, std::size_t queries, std::size_t dimension, std::uint32_t seed);
Dataset load_dataset(const std::string& base, const std::string& queries, const std::string& truth,
                     std::size_t query_limit);
std::vector<Neighbor> exact(std::span<const float> base, std::span<const float> query, std::size_t k);
}
