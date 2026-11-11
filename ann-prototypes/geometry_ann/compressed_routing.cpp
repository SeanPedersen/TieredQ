// TieredQ L2 model trained on the leaf centroids themselves; search_fast scores -|q - c|^2 on compressed codes.
#include "compressed_routing.h"
#include <tieredq/tieredq.h>
#include <omp.h>
#include <stdexcept>
namespace geometry_ann {
CompressedRouter::CompressedRouter(std::span<const float> centroids, std::size_t dimension, std::size_t bits_per_dimension,
                                   std::size_t scan_bits, std::uint32_t seed) {
    if (!dimension || centroids.empty() || centroids.size() % dimension || !bits_per_dimension || bits_per_dimension > 8)
        throw std::invalid_argument("compressed routing needs centroids and 1..8 bits per dimension");
    if (scan_bits > 8) throw std::invalid_argument("routing scan bits must be 0..8");
    auto rows = centroids.size() / dimension;
    auto config = tq::TieredQConfig::target_code_bits(std::uint32_t(dimension), std::uint32_t(bits_per_dimension * dimension));
    config.metric = tq::TieredQMetric::l2;
    config.max_scan_bits = std::uint32_t(scan_bits);
    auto codec = tq::TieredQ::train(config, centroids.data(), rows, seed);
    index_ = std::make_unique<tq::TieredQIndex>(std::move(codec), tq::TieredQIndexMode::fast);
    index_->add(centroids.data(), rows);
}
CompressedRouter::~CompressedRouter() = default;
CompressedRouter::CompressedRouter(CompressedRouter&&) noexcept = default;
CompressedRouter& CompressedRouter::operator=(CompressedRouter&&) noexcept = default;
std::vector<RankedLeaf> CompressedRouter::rank(std::span<const float> query, std::size_t candidates) const {
    auto hits = index_->search_fast(query.data(), candidates);
    std::vector<RankedLeaf> result; result.reserve(hits.size());
    for (const auto& hit : hits) result.push_back({-double(hit.score), hit.id});
    return result;
}
std::size_t CompressedRouter::bytes() const { return index_->storage_bytes() + index_->codec().memory_bytes(); }
std::size_t CompressedRouter::code_bits() const { return index_->codec().config().code_bits; }
void set_router_threads(std::size_t threads) { omp_set_num_threads(int(threads)); }
}
