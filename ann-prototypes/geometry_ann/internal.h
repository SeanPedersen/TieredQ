// Shared training and geometry contracts; implementation details stay outside the public API.
#pragma once
#include "index.h"
#include "matrix.h"
#include <random>
namespace geometry_ann {
using Group = std::vector<Id>;
double sq8_l2(std::span<const float> query, std::span<const std::int8_t> code, std::span<const float> scales, std::span<const float> offsets = {});
double sq8_lower_bound(std::span<const float> query, std::span<const std::int8_t> code, std::span<const float> scales, std::span<const float> offsets = {});
double centroid_l2(std::span<const float> a, std::span<const float> b);
double scan_lower_bound(std::span<const float> a, std::span<const float> b);
double l2(std::span<const float> a, std::span<const float> b);
std::span<const float> row(std::span<const float> data, std::size_t d, Id id);
std::vector<Group> cluster(Matrix data, std::size_t d, const Group& ids,
                           std::size_t count, const BuildOptions& options, std::mt19937& rng);
// Lifts the hierarchical restriction at coarse boundaries; returns how many vectors changed leaf.
// Leaves that become empty are removed together with their `owners` entry.
std::size_t refine_leaves(Matrix data, std::size_t d, std::vector<Group>& leaves, std::vector<std::size_t>& owners,
                          std::span<const float> parent_centroids, const BuildOptions& options);
// Leaves from per-row labels (empty labels dropped); returns parent member groups and sets
// `owners[leaf]` from clustering the leaf centroids into `options.coarse` parents.
std::vector<Group> external_leaves(Matrix data, std::size_t d, const BuildOptions& options, std::mt19937& rng,
                                   std::vector<Group>& leaves, std::vector<std::size_t>& owners);
std::vector<Group> adaptive(Matrix data, std::size_t d,
                            std::vector<Group> coarse, const BuildOptions& options,
                            std::vector<std::size_t>& owners, std::mt19937& rng);
Envelope centroid_envelope(Matrix data, std::size_t d, const Group& ids);
void add_radius(Envelope& result, Matrix data, std::size_t d, const Group& ids);
void add_projections(Envelope& result, Matrix data, std::size_t d, const Group& ids,
                     std::size_t count, std::size_t sample_size, std::mt19937& rng);
std::vector<EdgeSupport> build_edge_supports(Matrix packed, std::size_t d,
    std::span<const Leaf> leaves, std::span<const Id> edges, std::size_t degree);
double edge_support_lower_bound(std::span<const EdgeSupport> supports,
    std::span<const Id> neighbors, std::span<const double> distances,
    double centroid_distance, std::size_t dimension);
struct GraphTopology {
    std::vector<Id> edges, entries;
    std::size_t degree = 0, distance_evaluations = 0;
};
struct GraphView { std::span<const Id> edges, entries; std::size_t degree; };
GraphTopology build_graph(std::span<const Leaf> leaves, std::span<const Parent> parents, const BuildOptions& options);
Envelope envelope(Matrix data, std::size_t d, const Group& ids,
                  std::size_t directions, std::size_t sample, std::mt19937& rng, bool statistics = true);
struct BoundValues { double radius, max_slab, joint; };
BoundValues distance_bounds(std::span<const float> q, const Envelope& geometry, bool projections, double centroid_distance = -1);
double lower_bound(std::span<const float> q, const Envelope& geometry, bool projections, double centroid_distance = -1);
struct SharedQuery { std::vector<double> low, high; };
void add_shared_ranges(Envelope& leaf, const Envelope& basis, Matrix data, std::size_t d, const Group& ids);
SharedQuery shared_query(std::span<const float> query, const Envelope& basis);
double shared_lower_bound(const SharedQuery& query, const Envelope& basis, const Envelope& leaf);
double shared_residual_lower_bound(const SharedQuery& query, const Envelope& basis, const Envelope& leaf, double centroid_distance);
// Leaf owning a packed row; leaves are contiguous and sorted by offset.
std::size_t leaf_containing(std::span<const Leaf> leaves, std::size_t packed);
// cost[n]: minimum whole-leaf population covering at least n truth IDs (exact knapsack-style DP).
std::vector<std::size_t> cover_costs(std::span<const std::size_t> profits, std::span<const Leaf> leaves, std::size_t k);

}
