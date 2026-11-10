// Convex cell lower bounds: member-extreme slabs along centroid-difference directions,
// their joint (polytope) distance, and the convex-hull distance ceiling.
#pragma once
#include "internal.h"
#include <optional>
namespace geometry_ann {
// Per leaf l and direction a_e = c_e - c_l toward its e-th nearest centroid:
// [low, high] encloses (x - c_l)·a_e for every member x. Gram holds a_e·a_f.
struct CellSlabs {
    std::size_t directions = 0;
    std::vector<Id> neighbors;
    std::vector<double> low, high, gram;
    std::size_t bytes() const;
};
CellSlabs build_cell_slabs(Matrix packed, std::size_t d, std::span<const Leaf> leaves,
                           std::size_t directions, std::size_t threads);
struct SlabBounds { double single = 0, joint = 0; };
// Uses the leading `used` directions. Query projections come from full flat centroid distances.
// `sweeps` bounds the coordinate-ascent passes; zero computes only the single-slab bound.
SlabBounds slab_bounds(const CellSlabs& slabs, std::size_t leaf, std::size_t used,
                       std::span<const double> centroid_distances, std::size_t d, std::size_t sweeps);
struct HullDecision { double lower = 0, upper = 0; std::size_t iterations = 0; };
struct HullStop {
    std::optional<double> threshold;   // stop once the distance is certified on either side
    double relative_gap = 0;           // stop once upper - lower <= relative_gap * upper
    std::size_t max_iterations = 64;
};
// Gilbert iterations on the decoded members of one leaf: lower/upper bracket the squared
// distance from the query to their convex hull.
HullDecision hull_distance(std::span<const float> query, Matrix packed, std::size_t d,
                           std::size_t offset, std::size_t size, std::span<const double> member_distances,
                           const HullStop& stop);
}
