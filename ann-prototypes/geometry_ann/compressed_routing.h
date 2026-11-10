// Leaf-centroid ranking over a TieredQ-compressed centroid table (cache-resident at low bits per dimension).
// The table only orders leaves; callers re-rank the head of the order with exact centroid distances.
#pragma once
#include "index.h"
#include <memory>
namespace dv { class TieredQIndex; }
namespace geometry_ann {
struct RankedLeaf { double distance; Id leaf; };
class CompressedRouter {
public:
    CompressedRouter(std::span<const float> centroids, std::size_t dimension, std::size_t bits_per_dimension,
                     std::size_t scan_bits, std::uint32_t seed);
    ~CompressedRouter();
    CompressedRouter(CompressedRouter&&) noexcept;
    CompressedRouter& operator=(CompressedRouter&&) noexcept;
    // Approximate squared distances of the `candidates` nearest leaves, ascending.
    std::vector<RankedLeaf> rank(std::span<const float> query, std::size_t candidates) const;
    std::size_t bytes() const;
    std::size_t code_bits() const;
private:
    std::unique_ptr<dv::TieredQIndex> index_;
};
// The TieredQ scan parallelises single queries over OpenMP threads; benchmarks set one thread after the build.
void set_router_threads(std::size_t threads);
}
