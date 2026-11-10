// Graph connectivity, bounded-search equivalence, and far-bridge safety regressions.
#include "search_state.h"
#include "graph_experiment.h"
#include "restricted_assignment.h"
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
// With every candidate re-ranked exactly, compressed routing reproduces flat routing; without re-ranking it
// still honours every cap and returns a full top-k.
// FP32 routing distances keep the exhaustive flat order on well-separated data and reject geometric bounds.
void float_routing_test(const Index& index, std::span<const float> data, std::size_t dimension) {
    for (std::size_t q = 0; q < 8; ++q) {
        auto query = data.subspan(q * dimension, dimension);
        SearchOptions exact; exact.routing = Routing::flat; exact.bounds = BoundsMode::none; exact.ordering = Ordering::centroid;
        exact.leaf_scans = index.leaf_count();
        auto fast = exact; fast.float_routing = true;
        auto a = index.search(query, exact), b = index.search(query, fast);
        for (std::size_t k = 0; k < a.neighbors.size(); ++k) require(a.neighbors[k].id == b.neighbors[k].id, "float routing changed exhaustive results");
    }
    SearchOptions bounded; bounded.float_routing = true; bounded.bounds = BoundsMode::radius;
    bool rejected = false;
    try { index.search(data.first(dimension), bounded); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "float routing accepted geometric bounds");
}
void compressed_routing_test() {
    // TieredQ trains on at least 2 x 256 rows, so the routing table needs that many leaves.
    constexpr std::size_t population = 12000, dimension = 16;
    std::mt19937 rng(23); std::normal_distribution<float> normal;
    std::vector<float> data(population * dimension);
    for (auto& value : data) value = normal(rng);
    BuildOptions build; build.variant = Variant::fixed; build.coarse = 8; build.children = 80; build.leaf_budget = 640;
    build.routing_bits = 2; build.routing_scan_bits = 4;
    Index index(data, dimension, build);
    require(index.router_bytes() > 0, "routing table missing");
    for (std::size_t q = 0; q < 8; ++q) {
        auto query = std::span(data).subspan(q * dimension, dimension);
        SearchOptions flat; flat.routing = Routing::flat; flat.bounds = BoundsMode::none; flat.ordering = Ordering::centroid;
        flat.leaf_scans = index.leaf_count(); flat.vector_scans = 700; flat.patience = 3;
        auto compressed = flat; compressed.routing = Routing::compressed;
        compressed.routing_candidates = index.leaf_count(); compressed.metadata_evaluations = index.leaf_count();
        auto expected = index.search(query, flat), found = index.search(query, compressed);
        require(found.vectors_scanned == expected.vectors_scanned && found.leaves_scanned == expected.leaves_scanned,
                "fully re-ranked compressed routing changed scan work");
        for (std::size_t k = 0; k < expected.neighbors.size(); ++k)
            require(found.neighbors[k].id == expected.neighbors[k].id, "fully re-ranked compressed routing changed results");
        auto lazy = compressed; lazy.routing_initial = 4; lazy.metadata_evaluations = 3;
        auto single = lazy; single.routing_initial = 0;
        auto grown = index.search(query, lazy), once = index.search(query, single);
        require(grown.vectors_scanned == once.vectors_scanned && grown.leaves_scanned == once.leaves_scanned,
                "lazy ranking changed scan work");
        for (std::size_t k = 0; k < once.neighbors.size(); ++k)
            require(grown.neighbors[k].id == once.neighbors[k].id, "lazy ranking changed results");
        require(grown.routing_rounds > 1 && once.routing_rounds == 1, "lazy ranking did not grow");
        compressed.metadata_evaluations = 0; compressed.routing_candidates = 20; compressed.patience = 0; compressed.vector_scans = 300;
        found = index.search(query, compressed);
        require(found.neighbors.size() == compressed.k && found.metadata_evaluations == 0, "unranked compressed routing failed");
        require(found.vectors_scanned <= 300 && found.leaves_scanned <= 20, "compressed routing exceeded a cap");
    }
}
// Probing every super-centroid is exhaustive; probing a few scores a fraction of the centroids and stays nearly exact.
void restricted_assignment_test() {
    constexpr std::size_t centroids = 300, rows = 6000, dimension = 16;
    std::mt19937 rng(31); std::normal_distribution<float> normal;
    std::vector<float> centers(centroids * dimension), data(rows * dimension);
    for (auto& value : centers) value = normal(rng);
    for (std::size_t r = 0; r < rows; ++r) for (std::size_t j = 0; j < dimension; ++j)
        data[r * dimension + j] = centers[(r % centroids) * dimension + j] + 0.6f * normal(rng);
    DecodeRows decode = [&](std::size_t first, std::size_t count, std::span<float> out) {
        std::copy_n(data.begin() + std::ptrdiff_t(first * dimension), count * dimension, out.begin());
    };
    auto exact = brute_force_assign(data, centers, dimension, 2);
    AssignmentOptions all; all.supers = 12; all.probes = 12; all.threads = 2; all.chunk_rows = 1000;
    std::vector<std::uint32_t> labels(rows);
    restricted_assign(rows, centers, dimension, all, decode, labels);
    require(labels == exact, "probing every super-centroid differs from brute force");
    AssignmentOptions few = all; few.probes = 4;
    auto stats = restricted_assign(rows, centers, dimension, few, decode, labels);
    std::size_t same = 0;
    for (std::size_t r = 0; r < rows; ++r) same += labels[r] == exact[r];
    require(double(same) / rows >= 0.97, "restricted assignment is not nearly exact");
    require(stats.mean_candidates < 0.5 * centroids, "restricted assignment scored too many centroids");
}
// Patience only shortens scans: a window longer than the leaf count reproduces the exhaustive
// result, scan work grows monotonically with the window, and caps still bound every query.
void patience_test(const Index& index, std::span<const float> data, std::size_t dimension) {
    for (std::size_t q = 0; q < 12; ++q) {
        auto query = data.subspan(q * dimension, dimension);
        SearchOptions exhaustive; exhaustive.routing = Routing::flat; exhaustive.bounds = BoundsMode::none;
        exhaustive.ordering = Ordering::centroid; exhaustive.leaf_scans = index.leaf_count();
        auto expected = index.search(query, exhaustive);
        auto unlimited = exhaustive; unlimited.patience = index.leaf_count() + 1;
        auto same = index.search(query, unlimited);
        require(!same.stopped_by_patience && same.vectors_scanned == expected.vectors_scanned, "long patience changed scan work");
        for (std::size_t k = 0; k < expected.neighbors.size(); ++k)
            require(same.neighbors[k].id == expected.neighbors[k].id, "long patience changed results");
        std::size_t previous = 0;
        for (std::size_t window : {1, 2, 4, 8}) {
            auto options = exhaustive; options.patience = window;
            auto result = index.search(query, options);
            require(result.neighbors.size() == options.k, "patience stopped before filling top-k");
            require(result.vectors_scanned >= previous, "patience work is not monotone in its window");
            require(result.stopped_by_patience || result.leaves_scanned == index.leaf_count(), "scan ended without a stop reason");
            previous = result.vectors_scanned;
            options.vector_scans = 9;
            require(index.search(query, options).vectors_scanned <= 9, "patience bypassed the vector cap");
        }
    }
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
    auto threaded_build = build; threaded_build.build_threads = 4;
    auto threaded = build_graph(leaves, parents, threaded_build);
    require(threaded.edges == graph.edges && threaded.entries == graph.entries &&
            threaded.distance_evaluations == graph.distance_evaluations, "threaded graph build differs");
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
    patience_test(index, data, dimension);
    float_routing_test(index, data, dimension);
    compressed_routing_test();
    restricted_assignment_test();
}
}
