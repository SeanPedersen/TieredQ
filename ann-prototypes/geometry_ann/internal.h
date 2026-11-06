// Shared training and geometry contracts; implementation details stay outside the public API.
#pragma once
#include "index.h"
#include <random>
namespace geometry_ann {
using Group = std::vector<Id>;
double l2(std::span<const float> a, std::span<const float> b);
std::span<const float> row(std::span<const float> data, std::size_t d, Id id);
std::vector<Group> cluster(std::span<const float> data, std::size_t d, const Group& ids,
                           std::size_t count, const BuildOptions& options, std::mt19937& rng);
std::vector<Group> adaptive(std::span<const float> data, std::size_t d,
                            std::vector<Group> coarse, const BuildOptions& options,
                            std::vector<std::size_t>& owners, std::mt19937& rng);
Envelope centroid_envelope(std::span<const float> data, std::size_t d, const Group& ids);
void add_radius(Envelope& result, std::span<const float> data, std::size_t d, const Group& ids);
void add_projections(Envelope& result, std::span<const float> data, std::size_t d, const Group& ids,
                     std::size_t count, std::size_t sample_size, std::mt19937& rng);
struct GraphTopology {
    std::vector<Id> edges, entries;
    std::size_t degree = 0, distance_evaluations = 0;
};
struct GraphView { std::span<const Id> edges, entries; std::size_t degree; };
GraphTopology build_graph(std::span<const Leaf> leaves, std::span<const Parent> parents, const BuildOptions& options);
Envelope envelope(std::span<const float> data, std::size_t d, const Group& ids,
                  std::size_t directions, std::size_t sample, std::mt19937& rng, bool statistics = true);
struct BoundValues { double radius, max_slab, joint; };
BoundValues distance_bounds(std::span<const float> q, const Envelope& geometry, bool projections, double centroid_distance = -1);
double lower_bound(std::span<const float> q, const Envelope& geometry, bool projections, double centroid_distance = -1);
struct SharedQuery { std::vector<double> low, high; };
void add_shared_ranges(Envelope& leaf, const Envelope& basis, std::span<const float> data, std::size_t d, const Group& ids);
SharedQuery shared_query(std::span<const float> query, const Envelope& basis);
double shared_lower_bound(const SharedQuery& query, const Envelope& basis, const Envelope& leaf);

}
