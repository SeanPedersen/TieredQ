// Offline simulations of scan policies that change which leaves must be scanned:
// boundary spilling (duplicate vectors near a neighbour's bisector) and adaptive stopping.
#pragma once
#include "internal.h"
#include <limits>
namespace geometry_ann {
inline constexpr std::uint32_t no_spill = std::numeric_limits<std::uint32_t>::max();
// Row-major lists of the `count` nearest other leaves by centroid distance, nearest first.
std::vector<Id> nearest_leaves(std::span<const Leaf> leaves, std::size_t count, std::size_t threads);
// Per packed row: best alternative leaf among its leaf's candidates, and the relative margin
// (d_alt - d_own) / d_own. A small margin means the row lies near the bisector hyperplane.
struct SpillPlan {
    std::vector<std::uint32_t> leaf_of_row, alternative;
    std::vector<float> margin;
};
SpillPlan build_spill_plan(Matrix packed, std::size_t d, std::span<const Leaf> leaves,
                           std::span<const Id> candidates, std::size_t candidate_count, std::size_t threads);
// Rows with margin below tau are also stored in their alternative leaf.
struct Spill {
    std::vector<std::uint32_t> target;      // per row, or no_spill
    std::vector<std::size_t> leaf_sizes;    // sizes including spilled-in copies
    double fraction = 0;                    // extra vectors / original vectors
};
Spill apply_spill(const SpillPlan& plan, std::span<const Leaf> leaves, double tau);
// Whole leaves in `order` under a vector cap (leaves that no longer fit are skipped); a truth
// row counts when its own or its spill leaf is scanned.
double spilled_coverage(std::span<const std::size_t> order, const Spill& spill, std::span<const std::uint32_t> leaf_of_row,
                        std::span<const std::size_t> truth_rows, std::size_t budget);
// Scan leaves in `order` until `patience` consecutive leaves leave the current top-k unchanged
// or `max_vectors` would be exceeded. Coverage counts truth rows inside scanned leaves.
struct AdaptiveScan { std::size_t vectors = 0, leaves = 0; double coverage = 0; };
AdaptiveScan patience_scan(std::span<const std::size_t> order, std::span<const Leaf> leaves,
                           std::span<const double> member_distances, std::span<const std::size_t> truth_leaves,
                           std::size_t k, std::size_t patience, std::size_t max_vectors);
}
