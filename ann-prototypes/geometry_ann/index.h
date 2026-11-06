// Public contracts for immutable, packed two-level L2 indexes.
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <limits>
#include <vector>
namespace geometry_ann {
using Id = std::uint32_t;
enum class Variant { fixed, radius, pca };
enum class Ordering { centroid, bound, variance, hybrid };
enum class GraphScanOrder { routing, centroid };
enum class Routing { hierarchy, flat, graph };
enum class BoundsMode { automatic, none, radius, pca_max, pca, shared };
struct BuildStatistics {
    double clustering_s = 0, centroids_s = 0, radius_s = 0, projections_s = 0, packing_s = 0, graph_s = 0;
    double shared_basis_s = 0, shared_ranges_s = 0;
    std::size_t graph_distance_evaluations = 0;
};
struct BuildOptions {
    Variant variant = Variant::fixed;
    std::size_t coarse = 16, children = 32, leaf_budget = 512, target_leaf_size = 64;
    std::size_t iterations = 4, training_sample = 4096, pca_sample = 256, directions = 3;
    std::uint32_t seed = 42;
    bool local_kmeans_split = false;
    std::size_t graph_degree = 0, graph_parent_probes = 4, graph_refinements = 2;
    std::size_t shared_directions = 0, shared_sample = 4096;
};
struct SearchOptions {
    std::size_t k = 10, leaf_scans = 32, metadata_evaluations = 4096;
    Ordering ordering = Ordering::bound;
    std::size_t vector_scans = std::numeric_limits<std::size_t>::max();
    Routing routing = Routing::hierarchy;
    BoundsMode bounds = BoundsMode::automatic;
    double bound_weight = 0.5;
    std::size_t graph_seeds = 4;
    GraphScanOrder graph_scan_order = GraphScanOrder::centroid;
};
struct Neighbor {
    double distance;
    Id id;
    bool operator<(const Neighbor& other) const {
        return distance < other.distance || (distance == other.distance && id < other.id);
    }
};
struct SearchResult {
    std::vector<Neighbor> neighbors;
    std::size_t leaves_scanned = 0, vectors_scanned = 0, metadata_evaluations = 0, pruned = 0;
    bool metadata_exhausted = false;
    std::size_t leaves_skipped_for_vector_budget = 0;
    std::size_t parents_evaluated = 0, radius_evaluations = 0, projection_evaluations = 0, graph_expansions = 0;
    std::size_t positive_bounds = 0, projection_improvements = 0, routing_ns = 0, scanning_ns = 0;
    std::size_t shared_query_projections = 0, shared_interval_evaluations = 0;
};
struct Envelope {
    std::vector<float> centroid, directions;
    std::vector<double> low, high, norm, shared_low, shared_high;
    double radius = 0, variance = 0, projection_operator_norm_sq = 0;
};
struct Leaf { Envelope geometry; std::size_t offset = 0, size = 0; };
struct Parent { Envelope geometry; std::vector<std::size_t> leaves; };
class Index {
public:
    Index(std::span<const float> vectors, std::size_t dimension, const BuildOptions& options);
    SearchResult search(std::span<const float> query, const SearchOptions& options) const;
    std::size_t raw_bytes() const;
    std::size_t overhead_bytes() const;
    std::size_t leaf_count() const { return leaves_.size(); }
    const BuildStatistics& build_statistics() const { return statistics_; }
    std::size_t graph_bytes() const { return (graph_edges_.capacity() + graph_entries_.capacity()) * sizeof(Id); }
    std::size_t bounds_bytes(BoundsMode mode) const;
private:
    SearchResult search_hierarchy(std::span<const float> query, const SearchOptions& options) const;
    SearchResult search_graph(std::span<const float> query, const SearchOptions& options) const;
    std::size_t dimension_;
    Variant variant_;
    std::vector<float> vectors_;
    std::vector<Id> ids_;
    std::vector<Leaf> leaves_;
    std::vector<Parent> parents_;
    std::vector<Id> graph_edges_, graph_entries_;
    std::size_t graph_degree_ = 0;
    BuildStatistics statistics_;
    Envelope shared_geometry_;
};
}
