// Statistical leaf-ranking scores: expected nearest-member squared distance under an
// extreme-value model of member projections along the query direction. Not lower bounds.
#pragma once
#include "internal.h"
namespace geometry_ann {
struct SpreadModel {
    std::size_t axes = 0, dimension = 0;
    std::vector<double> basis;                             // axes × d, orthonormal rows
    std::vector<double> center_projection, axis_variance;  // leaf × axes
    std::vector<double> mean_radius_sq, log_size;          // per leaf
    std::size_t bytes() const;
};
// Shared basis from sampled PCA; per-leaf second moments over every member.
SpreadModel build_spread_model(Matrix packed, std::size_t d, std::span<const Leaf> leaves, std::size_t axes,
                               std::size_t sample, std::uint32_t seed, std::size_t threads);
std::vector<double> project_query(const SpreadModel& model, std::span<const float> query);
// Score = D + r̄² − 2β·√D·s·√(2 ln n), with D the centroid distance and s² the leaf variance
// along the query direction (isotropic r̄²/d unless `directional`).
double expected_nearest(const SpreadModel& model, std::size_t leaf, double centroid_distance,
                        std::span<const double> query_projection, double beta, bool directional);
}
