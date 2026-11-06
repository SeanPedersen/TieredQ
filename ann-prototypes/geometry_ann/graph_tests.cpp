// Graph connectivity, bounded-search equivalence, and far-bridge safety regressions.
#include "search_state.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
namespace geometry_ann {
namespace {
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void bridge_test() {
    std::vector<float> data = {10, 1000, 0}, query = {0};
    std::vector<Id> ids = {0, 1, 2}, edges = {1, 2, 0}, entries = {0};
    std::vector<Leaf> leaves;
    for (std::size_t i = 0; i < data.size(); ++i) {
        Envelope envelope; envelope.centroid = {data[i]}; leaves.push_back({envelope, i, 1});
    }
    Parent parent; parent.geometry.centroid = {10}; parent.leaves = {0, 1, 2};
    std::vector<Parent> parents = {parent};
    SearchOptions options; options.k = 1; options.leaf_scans = 3; options.bounds = BoundsMode::radius;
    SearchState state(query, options, Variant::radius, data, ids, leaves);
    state.scan({100, 100, 100, 0, false});
    auto visits = discover_graph(state, parents, leaves, {edges, entries, 1}, 1);
    for (auto visit : visits) if (state.can_scan()) state.scan(visit);
    auto result = state.finish();
    require(result.neighbors.front().id == 2, "far bridge prevented discovery of nearer leaf");
    require(result.graph_expansions == 3, "leaf pruning cut a graph path");
}
}
void check_graph() {
    constexpr std::size_t population = 181, dimension = 7;
    std::mt19937 rng(19); std::normal_distribution<float> normal;
    std::vector<float> data(population * dimension);
    for (auto& value : data) value = normal(rng);
    BuildOptions build; build.variant = Variant::pca; build.coarse = 4; build.leaf_budget = 24;
    build.target_leaf_size = 4; build.shared_directions = 4; build.graph_degree = 8; build.local_kmeans_split = true;
    Index index(data, dimension, build);
    for (std::size_t q = 0; q < 12; ++q) {
        auto query = std::span(data).subspan(q * dimension, dimension);
        SearchOptions exact; exact.routing = Routing::flat; exact.leaf_scans = index.leaf_count();
        auto expected = index.search(query, exact);
        for (auto bounds : {BoundsMode::none, BoundsMode::radius, BoundsMode::pca_max, BoundsMode::pca, BoundsMode::shared})
            for (auto order : {Ordering::centroid, Ordering::hybrid, Ordering::bound}) {
                auto options = exact; options.routing = Routing::graph;
                options.bounds = bounds; options.ordering = order;
                options.graph_scan_order = order == Ordering::bound ? GraphScanOrder::routing : GraphScanOrder::centroid;
                auto result = index.search(query, options);
                require(result.neighbors.size() == expected.neighbors.size(), "graph result count mismatch");
                for (std::size_t k = 0; k < result.neighbors.size(); ++k)
                    require(result.neighbors[k].id == expected.neighbors[k].id, "complete graph search differs from exhaustive");
                options.metadata_evaluations = 11; options.vector_scans = 17; options.leaf_scans = 2;
                result = index.search(query, options);
                require(result.metadata_evaluations <= 11 && result.vectors_scanned <= 17 && result.leaves_scanned <= 2,
                        "graph search exceeded a hard budget");
            }
    }
    std::vector<Leaf> leaves(17); std::vector<Parent> parents(2);
    for (std::size_t i = 0; i < leaves.size(); ++i) {
        leaves[i].geometry.centroid = {float(i), float(i % 3)}; parents[i % parents.size()].leaves.push_back(i);
    }
    for (auto& parent : parents) parent.geometry.centroid = {0, 0};
    auto graph = build_graph(leaves, parents, build);
    std::vector<bool> seen(leaves.size(), false); Group frontier = {0}; seen[0] = true;
    while (!frontier.empty()) {
        Id node = frontier.back(); frontier.pop_back(); Group neighbors;
        for (std::size_t e = 0; e < graph.degree; ++e) {
            Id next = graph.edges[node * graph.degree + e];
            require(next < leaves.size() && next != node, "invalid graph endpoint");
            require(std::find(neighbors.begin(), neighbors.end(), next) == neighbors.end(), "duplicate graph edge");
            neighbors.push_back(next);
            if (!seen[next]) { seen[next] = true; frontier.push_back(next); }
        }
    }
    require(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }), "centroid graph disconnected");
    Envelope box; box.centroid = {0, 0}; box.radius = 100;
    box.directions = {1, 0, 0, 1}; box.norm = {1, 1}; box.low = {-1, -1}; box.high = {1, 1};
    box.projection_operator_norm_sq = 1;
    std::vector<float> outside = {3, 3};
    auto bounds = distance_bounds(outside, box, true);
    require(bounds.joint > 7.99 && bounds.joint <= 8 && bounds.max_slab <= 4, "joint projection metric incorrect");
    box.directions = {1, 0, 1, 0}; box.projection_operator_norm_sq = 2;
    bounds = distance_bounds(outside, box, true);
    require(bounds.joint <= 4 && bounds.joint > 3.99, "nonorthogonal projections were over-counted");
    Group all(population); std::iota(all.begin(), all.end(), Id(0));
    Group subset(all.begin(), all.begin() + population / 3);
    auto basis = envelope(data, dimension, all, 4, 47, rng);
    auto leaf = centroid_envelope(data, dimension, subset);
    add_shared_ranges(leaf, basis, data, dimension, subset);
    for (Id id : subset)
        require(shared_lower_bound(shared_query(row(data, dimension, id), basis), basis, leaf) == 0,
                "shared envelope excludes a member");
    for (std::size_t trial = 0; trial < 100; ++trial) {
        std::vector<float> query(dimension);
        for (auto& value : query) value = normal(rng) * 20;
        double nearest = std::numeric_limits<double>::infinity();
        for (Id id : subset) nearest = std::min(nearest, l2(query, row(data, dimension, id)));
        require(shared_lower_bound(shared_query(query, basis), basis, leaf) <= nearest,
                "shared projection bound exceeds exact distance");
    }
    bridge_test();
}
}
