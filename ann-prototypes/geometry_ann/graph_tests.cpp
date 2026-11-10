// Graph connectivity, bounded-search equivalence, and far-bridge safety regressions.
#include "search_state.h"
#include "graph_experiment.h"
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
    BenchmarkOptions config; config.build = build;
    for (auto mode : {BoundsMode::none, BoundsMode::radius}) {
        config.selected_bounds = mode;
        auto lean_build = graph_build_options(config);
        require(lean_build.directions == 0 && lean_build.shared_directions == 0, "unused geometry retained");
        Index lean(data, dimension, lean_build);
        require(lean.overhead_bytes() < index.overhead_bytes(), "removing metadata did not reduce RAM");
        SearchOptions search; search.routing = Routing::graph; search.bounds = mode;
        search.metadata_evaluations = 16; search.vector_scans = 100;
        auto query = std::span(data).first(dimension);
        auto full_result = index.search(query, search), lean_result = lean.search(query, search);
        require(full_result.neighbors.size() == lean_result.neighbors.size(), "lean build lost neighbors");
        for (std::size_t i = 0; i < full_result.neighbors.size(); ++i)
            require(full_result.neighbors[i].id == lean_result.neighbors[i].id &&
                    full_result.neighbors[i].distance == lean_result.neighbors[i].distance, "metadata removal changed results");
    }
    config.variant = "V0"; config.selected_bounds = BoundsMode::none;
    auto fixed_build = graph_build_options(config);
    require(fixed_build.variant == Variant::fixed && fixed_build.shared_directions == 0, "V0 graph build is not fixed");
    config.selected_bounds = BoundsMode::radius;
    auto radius_build=graph_build_options(config);
    require(radius_build.variant==Variant::fixed && radius_build.geometry_metadata,
            "fixed geometry build changed partitions or omitted metadata");
    Index fixed(data, dimension, fixed_build);
    SearchOptions complete; complete.routing = Routing::graph; complete.bounds = BoundsMode::none;
    complete.leaf_scans = fixed.leaf_count(); complete.metadata_evaluations = 10000;
    auto fixed_query = std::span(data).first(dimension);
    auto fixed_graph = fixed.search(fixed_query, complete);
    complete.routing = Routing::flat;
    auto fixed_flat = fixed.search(fixed_query, complete);
    require(fixed_graph.neighbors.size() == fixed_flat.neighbors.size(), "V0 graph is disconnected");
    for (std::size_t i = 0; i < fixed_flat.neighbors.size(); ++i)
        require(fixed_flat.neighbors[i].id == fixed_graph.neighbors[i].id, "complete V0 graph differs from flat");
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
        require(shared_residual_lower_bound(shared_query(query,basis),basis,leaf,l2(query,leaf.centroid))<=nearest,
                "shared residual bound exceeds exact distance");
        require(shared_lower_bound(shared_query(query, basis), basis, leaf) <= nearest,
                "shared projection bound exceeds exact distance");
    }
    // Residual information distinguishes a long thin cluster when projection
    // intervals and its enclosing ball both give zero lower bound.
    {
        std::vector<float> points={100,1,-100,1,100,-1,-100,-1};
        Group members={0,1,2,3}; Envelope axis; axis.centroid={0,0};
        axis.directions={1,0}; axis.norm={1}; axis.projection_operator_norm_sq=1.000000000001;
        auto thin=centroid_envelope(points,2,members); add_radius(thin,points,2,members);
        add_shared_ranges(thin,axis,points,2,members);
        std::vector<float> q={0,5}; auto projected=shared_query(q,axis);
        require(shared_lower_bound(projected,axis,thin)==0 && lower_bound(q,thin,false)==0,
                "thin-cluster baseline unexpectedly informative");
        auto residual=shared_residual_lower_bound(projected,axis,thin,l2(q,thin.centroid));
        require(residual>15.99 && residual<=16,"residual geometry failed to recover missing separation");
        // Duplicated/nonorthogonal stored directions remain safe via operator bound.
        axis.directions={1,0,1,0};axis.norm={1,1};axis.projection_operator_norm_sq=2.000000000001;
        add_shared_ranges(thin,axis,points,2,members);
        for (float magnitude : {1e-20f,1.f,1e20f}) for (int trial=0;trial<100;++trial) {
            q={normal(rng)*magnitude,normal(rng)*magnitude};
            double nearest=std::numeric_limits<double>::infinity();
            for (Id id:members) nearest=std::min(nearest,l2(q,row(points,2,id)));
            require(shared_residual_lower_bound(shared_query(q,axis),axis,thin,l2(q,thin.centroid))<=nearest,
                    "nonorthogonal residual bound exceeds true distance");
        }
    }
    // Edge support queries must agree with direct projections while remaining
    // conservative at extreme scales and at a cluster member.
    {
        std::vector<float> points={-2,-1,-2,1,2,-1,2,1};
        std::vector<Leaf> clusters(2);
        clusters[0].geometry.centroid={-2,0};clusters[0].offset=0;clusters[0].size=2;
        clusters[1].geometry.centroid={2,0};clusters[1].offset=2;clusters[1].size=2;
        std::vector<Id> edges={1,0};auto supports=build_edge_supports(points,2,clusters,edges,1);
        for (std::size_t l=0;l<2;++l)
            require(supports[l].low_witness>=clusters[l].offset && supports[l].low_witness<clusters[l].offset+2 &&
                    supports[l].high_witness>=clusters[l].offset && supports[l].high_witness<clusters[l].offset+2,
                    "edge support witness is not a cluster member");
        std::vector<float> q={0,0};std::vector<double> ds={4,4};
        auto b=edge_support_lower_bound(std::span(supports).first(1),std::span(edges).first(1),ds,4,2);
        require(b>3.99 && b<=4,"edge support lost exact plane separation");
        for (float magnitude:{1e-20f,1.f,1e20f}) for (int trial=0;trial<100;++trial) {
            q={normal(rng)*magnitude,normal(rng)*magnitude};
            for (std::size_t l=0;l<2;++l)ds[l]=l2(q,clusters[l].geometry.centroid);
            for (std::size_t l=0;l<2;++l) {
                double nearest=std::numeric_limits<double>::infinity();
                for (std::size_t i=clusters[l].offset;i<clusters[l].offset+2;++i) nearest=std::min(nearest,l2(q,row(points,2,Id(i))));
                auto bound=edge_support_lower_bound(std::span(supports).subspan(l,1),std::span(edges).subspan(l,1),ds,ds[l],2);
                require(bound<=nearest,"edge support exceeds member distance");
            }
        }
        clusters[1].geometry.centroid=clusters[0].geometry.centroid;
        supports=build_edge_supports(points,2,clusters,edges,1); ds={1,1};
        require(edge_support_lower_bound(std::span(supports).first(1),std::span(edges).first(1),ds,1,2)==0,
                "duplicate centroids produced an invalid support plane");
    }
    bridge_test();
}
}
