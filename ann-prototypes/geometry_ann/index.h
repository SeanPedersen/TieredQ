// Public contracts for immutable, packed two-level L2 indexes.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <limits>
#include <vector>
namespace geometry_ann {
using Id = std::uint32_t;
enum class Variant { fixed, radius, pca };
enum class Ordering { centroid, bound, variance, hybrid };
enum class GraphScanOrder { routing, centroid };
enum class Routing { hierarchy, flat, graph, compressed };
enum class BoundsMode { automatic, none, radius, pca_max, pca, shared, shared_residual, support };
struct BuildStatistics {
    double clustering_s = 0, centroids_s = 0, radius_s = 0, projections_s = 0, packing_s = 0, graph_s = 0;
    double shared_basis_s = 0, shared_ranges_s = 0, support_s = 0, reassign_s = 0, router_s = 0;
    std::size_t reassigned_vectors = 0;
    std::size_t graph_distance_evaluations = 0;
};
struct BuildOptions {
    Variant variant = Variant::fixed;
    std::size_t coarse = 16, children = 32, leaf_budget = 512, target_leaf_size = 64;
    std::size_t iterations = 4, training_sample = 4096, pca_sample = 256, directions = 3;
    std::uint32_t seed = 42;
    std::size_t build_threads = 1;  // k-means assignment and graph build; results are identical for any count
    // Fixed variant: reassign every vector to the nearest leaf among the leaves of its
    // `reassign_parents` nearest coarse parents (0 disables), then recompute leaf means.
    std::size_t reassign_parents = 0, reassign_iterations = 2;
    // Fixed variant: externally trained flat partition, one leaf label per input row. Coarse parents
    // are then derived by clustering leaf centroids. The caller keeps the labels alive during the build.
    std::span<const std::uint32_t> assignments;
    bool local_kmeans_split = false, geometry_metadata = false, neighbor_support = false;
    std::size_t graph_degree = 0, graph_parent_probes = 4, graph_refinements = 2;
    std::size_t shared_directions = 0, shared_sample = 4096;
    // >0 also builds a TieredQ table over the leaf centroids at this many bits per dimension (Routing::compressed).
    std::size_t routing_bits = 0;
    // Per-coordinate scan width of that table (0: library default). At most 4 keeps every tier inside the x86
    // nibble-lookup kernel, about 20x faster for one query than the generic scan the default can fall back to.
    std::size_t routing_scan_bits = 0;
    // Graph construction scores centroid pairs with the FP32 SIMD kernel instead of the exact double kernel.
    bool float_graph_build = false;
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
    // Stop once this many consecutive scanned leaves leave a full top-k unchanged (0 disables).
    // Applies on top of the leaf and vector caps, which remain hard ceilings.
    std::size_t patience = 0;
    // Routing::compressed: leaves ranked by the compressed table (the scan never goes past them); the first
    // `metadata_evaluations` of them are re-ranked with exact centroid distances.
    std::size_t routing_candidates = 4096;
    // >0: rank only this many leaves first and re-rank 4x deeper (up to routing_candidates) whenever scanning
    // exhausts them, so routing depth follows the query's scan depth. Top-k lists are nested, so results equal
    // a single ranking of the final depth.
    std::size_t routing_initial = 0;
    // Centroid distances for routing (flat, graph and the exact re-rank head) use the FP32 SIMD kernel; bounds must be none.
    bool float_routing = false;
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
    bool metadata_exhausted = false, stopped_by_patience = false;
    std::size_t leaves_skipped_for_vector_budget = 0;
    std::size_t parents_evaluated = 0, radius_evaluations = 0, projection_evaluations = 0, graph_expansions = 0;
    std::size_t positive_bounds = 0, projection_improvements = 0, routing_ns = 0, scanning_ns = 0;
    std::size_t shared_query_projections = 0, shared_interval_evaluations = 0, routing_rounds = 0;
};
struct Envelope {
    std::vector<float> centroid, directions;
    std::vector<double> low, high, norm, shared_low, shared_high;
    std::vector<double> shared_center_low, shared_center_high;
    double residual_low = 0, residual_high = 0;
    double radius = 0, variance = 0, projection_operator_norm_sq = 0;
};
struct Leaf { Envelope geometry; std::size_t offset = 0, size = 0; };
struct Parent { Envelope geometry; std::vector<std::size_t> leaves; };
struct QuantizedInput {
    std::span<const std::int8_t> codes;
    std::span<const float> scales, offsets;
};
struct EdgeSupport { double low = 0, high = 0, length_squared = 0; Id low_witness = 0, high_witness = 0; };
struct GeometryDiagnostic {
    std::size_t query, positive, above_reference, unsafe_members;
    double weight, mean_truth_cluster_rank, max_truth_cluster_rank, mean_bound_ratio;
    BoundsMode mode;
    std::size_t truth_clusters = 0, oracle_vectors_9 = 0, oracle_vectors_10 = 0;
    std::size_t ranked_vectors_9 = 0, ranked_vectors_10 = 0;
    bool witness_ordering = false;
    std::size_t witness_evaluations = 0;
};
struct CellDiagnosticOptions {
    std::size_t directions = 32, small_directions = 8, hull_iterations = 64, threads = 1;
    std::size_t cheap_sweeps = 4, converged_sweeps = 256;
    std::size_t rank_iterations = 512, spread_axes = 8, spread_sample = 16384;
    double rank_gap = .01;
    bool slabs = true, hull = true, hull_rank = true, statistical = true;
    std::vector<double> betas = {0, .5, 1, 1.5, 2};
    std::vector<std::size_t> coverage_budgets = {35000, 176816, 442041};
    // Scan-policy simulations: boundary spill by bisector margin and patience-based stopping.
    bool policies = false;
    std::size_t policy_neighbors = 32;
    std::vector<double> spill_taus = {.02, .05, .1, .2, .4};
    std::vector<std::size_t> patience_windows = {2, 4, 8, 16, 32, 64, 128};
};
// One lower-bound family on one query. "Certified" leaves have bound <= reference distance:
// the minimum set an exact search using that bound alone could not rule out.
struct CellScore {
    std::string name;
    std::size_t certified_clusters = 0, certified_vectors = 0, undecided_clusters = 0, undecided_vectors = 0;
    std::size_t pure_vectors_10 = 0, hybrid_vectors_10 = 0, unsafe = 0, hull_iterations = 0;
    double mean_truth_rank = 0, bound_ns_per_leaf = 0, mean_truth_bound_ratio = 0;
    std::vector<double> coverage;  // truth fraction scanned within each coverage budget (pure order)
    double adaptive_vectors = 0, adaptive_leaves = 0, adaptive_coverage = 0, spill_fraction = 0;
    bool ranked = true, lower_bound = true;
};
struct CellDiagnostic {
    std::size_t query = 0, truth_clusters = 0, oracle_vectors_10 = 0;
    double reference = 0;
    std::vector<CellScore> scores;
};
struct CellStudy {
    double slab_build_s = 0, spread_build_s = 0;
    std::size_t slab_bytes = 0, spread_bytes = 0;
    std::vector<CellDiagnostic> queries;
};
class CompressedRouter;
class Index {
public:
    Index(std::span<const float> vectors, std::size_t dimension, const BuildOptions& options, QuantizedInput quantized = {});
    SearchResult search(std::span<const float> query, const SearchOptions& options) const;
    std::vector<GeometryDiagnostic> diagnose_geometry(std::span<const float> queries,
        std::span<const std::vector<Id>> truth) const;
    CellStudy diagnose_cells(std::span<const float> queries, std::span<const std::vector<Id>> truth,
        const CellDiagnosticOptions& options) const;
    std::size_t raw_bytes() const;
    bool quantized() const { return !codes_.empty(); }
    std::size_t overhead_bytes() const;
    std::size_t leaf_count() const { return leaves_.size(); }
    const BuildStatistics& build_statistics() const { return statistics_; }
    std::size_t graph_bytes() const { return (graph_edges_.capacity() + graph_entries_.capacity()) * sizeof(Id); }
    std::size_t bounds_bytes(BoundsMode mode) const;
    std::size_t router_bytes() const;
private:
    SearchResult search_hierarchy(std::span<const float> query, const SearchOptions& options) const;
    SearchResult search_graph(std::span<const float> query, const SearchOptions& options) const;
    SearchResult search_compressed(std::span<const float> query, const SearchOptions& options) const;
    std::size_t dimension_;
    Variant variant_;
    bool geometry_metadata_ = false;
    std::vector<float> vectors_, code_scales_, code_offsets_;
    std::vector<std::int8_t> codes_;
    std::vector<Id> ids_;
    std::vector<Leaf> leaves_;
    std::vector<Parent> parents_;
    std::vector<Id> graph_edges_, graph_entries_;
    std::vector<EdgeSupport> edge_supports_;
    std::size_t graph_degree_ = 0;
    BuildStatistics statistics_;
    Envelope shared_geometry_;
    std::shared_ptr<const CompressedRouter> router_;
};
}
