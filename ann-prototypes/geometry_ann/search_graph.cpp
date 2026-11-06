// Budgeted centroid-graph discovery followed by exact scans; leaf bounds never cut graph paths.
#include "search_state.h"
#include <algorithm>
#include <chrono>
namespace geometry_ann {
std::vector<Visit> discover_graph(SearchState& state, std::span<const Parent> parents,
                                  std::span<const Leaf> leaves, GraphView graph, std::size_t seeds) {
    std::vector<Visit> coarse;
    for (std::size_t p = 0; p < parents.size(); ++p) {
        auto visit = state.evaluate(parents[p].geometry, p, true, BoundsMode::none);
        if (!visit) break;
        coarse.push_back(*visit);
    }
    std::sort(coarse.begin(), coarse.end(), [](const Visit& a, const Visit& b) {
        return a.centroid_distance < b.centroid_distance || (a.centroid_distance == b.centroid_distance && a.index < b.index);
    });
    VisitQueue frontier;
    std::vector<Visit> discovered;
    std::vector<std::uint8_t> seen(leaves.size(), 0);
    auto discover = [&](Id leaf) {
        if (seen[leaf]) return;
        auto visit = state.evaluate(leaves[leaf].geometry, leaf, false);
        if (!visit) return;
        seen[leaf] = 1; discovered.push_back(*visit); frontier.push(*visit);
    };
    for (std::size_t i = 0; i < std::min(seeds, coarse.size()); ++i)
        discover(graph.entries[coarse[i].index]);
    while (!frontier.empty() && !state.result.metadata_exhausted) {
        auto visit = frontier.top(); frontier.pop(); ++state.result.graph_expansions;
        // Pruning a leaf's vectors cannot justify pruning its graph neighbors.
        for (std::size_t edge = 0; edge < graph.degree; ++edge)
            discover(graph.edges[visit.index * graph.degree + edge]);
    }
    return discovered;
}
SearchResult Index::search_graph(std::span<const float> query, const SearchOptions& options) const {
    auto start = std::chrono::steady_clock::now();
    SearchState state(query, options, variant_, vectors_, ids_, leaves_, &shared_geometry_);
    auto discovered = discover_graph(state, parents_, leaves_, {graph_edges_, graph_entries_, graph_degree_}, options.graph_seeds);
    std::sort(discovered.begin(), discovered.end(), [&](const Visit& a, const Visit& b) {
        if (options.graph_scan_order == GraphScanOrder::routing) return VisitGreater{}(b, a);
        return a.centroid_distance < b.centroid_distance || (a.centroid_distance == b.centroid_distance && a.index < b.index);
    });
    auto routed = std::chrono::steady_clock::now();
    for (const auto& visit : discovered) {
        if (!state.can_scan()) break;
        state.scan(visit);
    }
    auto result = state.finish();
    result.routing_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(routed - start).count();
    result.scanning_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - routed).count();
    return result;
}
}
