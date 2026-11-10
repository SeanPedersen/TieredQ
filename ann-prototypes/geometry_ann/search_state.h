// Shared query accounting and exact leaf scanning; traversal policies own only navigation.
#pragma once
#include "internal.h"
#include <optional>
#include <queue>
namespace geometry_ann {
struct Visit {
    double priority, bound, centroid_distance;
    std::size_t index;
    bool parent;
};
struct VisitGreater {
    bool operator()(const Visit& a, const Visit& b) const;
};
using VisitQueue = std::priority_queue<Visit, std::vector<Visit>, VisitGreater>;
class SearchState {
public:
    SearchState(std::span<const float> query, const SearchOptions& options, Variant variant,
                std::span<const float> vectors, std::span<const Id> ids, std::span<const Leaf> leaves, const Envelope* shared_geometry = nullptr, QuantizedInput quantized = {});
    std::optional<Visit> evaluate(const Envelope& geometry, std::size_t index, bool parent,
                                  BoundsMode override_mode = BoundsMode::automatic);
    bool discard_if_bounded(const Visit& visit);
    void scan(const Visit& visit);
    bool can_scan() const;
    SearchResult finish();
    SearchResult result;
private:
    std::span<const float> query_, vectors_;
    QuantizedInput quantized_;
    std::span<const Id> ids_;
    std::span<const Leaf> leaves_;
    const SearchOptions& options_;
    BoundsMode mode_;
    const Envelope* shared_geometry_;
    SharedQuery shared_query_;
    std::priority_queue<Neighbor> best_;
};
std::vector<Visit> discover_graph(SearchState& state, std::span<const Parent> parents,
                                  std::span<const Leaf> leaves, GraphView graph, std::size_t seeds);

}
