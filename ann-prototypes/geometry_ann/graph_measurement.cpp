// Query timing and work counters; settings are never tuned using these measurements internally.
#include "graph_experiment.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ostream>
#include <stdexcept>
namespace geometry_ann {
GraphObservation graph_measure(const Index& index, const Dataset& data, const BenchmarkOptions& config, const GraphSetting& setting) {
    SearchOptions options; options.graph_scan_order = setting.scan_order; options.bounds = setting.bounds; options.bound_weight = setting.weight;
    options.ordering = setting.weight == 0 ? Ordering::centroid : setting.weight == 1 ? Ordering::bound : Ordering::hybrid;
    options.routing = setting.routing == "graph" ? Routing::graph : setting.routing == "flat" ? Routing::flat : Routing::hierarchy;
    options.graph_seeds = config.graph_seeds; options.metadata_evaluations = setting.metadata_cap;
    options.vector_scans = setting.vector_cap; options.leaf_scans = config.leaf_scans.value_or(index.leaf_count());
    constexpr std::size_t warmup_queries = 20;
    for (std::size_t i = 0; i < std::min(warmup_queries, data.truth.size()); ++i)
        index.search(std::span(data.queries).subspan(i * data.dimension, data.dimension), options);
    GraphObservation result; std::vector<double> latency; std::size_t hits = 0;
    for (std::size_t q = 0; q < data.truth.size(); ++q) {
        auto start = std::chrono::steady_clock::now();
        auto found = index.search(std::span(data.queries).subspan(q * data.dimension, data.dimension), options);
        latency.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count());
        if (found.vectors_scanned > options.vector_scans || found.metadata_evaluations > options.metadata_evaluations ||
            found.leaves_scanned > options.leaf_scans) throw std::logic_error("query exceeded a hard budget");
        for (auto neighbor : found.neighbors)
            hits += std::find(data.truth[q].begin(), data.truth[q].end(), neighbor.id) != data.truth[q].end();
        result.vectors += found.vectors_scanned; result.metadata += found.metadata_evaluations;
        result.leaves += found.leaves_scanned; result.pruned += found.pruned; result.skipped += found.leaves_skipped_for_vector_budget;
        result.projections += found.projection_evaluations; result.radius += found.radius_evaluations;
        result.expansions += found.graph_expansions; result.positive += found.positive_bounds;
        result.improved += found.projection_improvements; result.routing_us += double(found.routing_ns) / 1000;
        result.scanning_us += double(found.scanning_ns) / 1000; result.exhausted += found.metadata_exhausted;
        result.shared_query_projections += found.shared_query_projections; result.shared_intervals += found.shared_interval_evaluations;
        result.max_vectors = std::max(result.max_vectors, found.vectors_scanned);
        result.max_metadata = std::max(result.max_metadata, found.metadata_evaluations);
    }
    double count = double(data.truth.size()); result.recall = double(hits) / (10 * count);
    for (auto field : {&GraphObservation::vectors, &GraphObservation::metadata, &GraphObservation::leaves,
                      &GraphObservation::pruned, &GraphObservation::skipped, &GraphObservation::projections,
                      &GraphObservation::radius, &GraphObservation::expansions, &GraphObservation::positive,
                      &GraphObservation::improved, &GraphObservation::routing_us, &GraphObservation::scanning_us,
                      &GraphObservation::exhausted, &GraphObservation::shared_query_projections, &GraphObservation::shared_intervals}) result.*field /= count;
    std::sort(latency.begin(), latency.end());
    result.p50 = latency[std::size_t(std::ceil(.50 * count)) - 1]; result.p99 = latency[std::size_t(std::ceil(.99 * count)) - 1];
    return result;
}
void graph_csv_header(std::ostream& out) {
    out << "routing,bounds,weight,metadata_cap,vector_cap,recall10,p50_us,p99_us,mean_vectors,max_vectors,mean_metadata,max_metadata,mean_leaves,mean_pruned,mean_skipped,mean_projection_components,mean_radius_evaluations,mean_graph_expansions,mean_positive_bounds,mean_projection_improvements,mean_routing_us,mean_scanning_us,metadata_exhaustion_rate,mean_scan_fraction,leaves,raw_bytes,full_overhead_bytes,graph_bytes,active_bounds_payload_bytes,build_s,clustering_s,centroids_s,radius_s,projections_s,packing_s,graph_s,graph_build_distances,nq,build_peak_rss_bytes,shared_basis_s,shared_ranges_s,mean_shared_query_projections,mean_shared_interval_checks,scan_order,vector_storage,original_fp32_bytes,support_build_s\n";
}
void graph_csv_row(std::ostream& out, const Index& index, const Dataset& data, const GraphSetting& s,
                   const GraphObservation& o, double build_seconds, std::size_t peak_rss) {
    const auto& b = index.build_statistics();
    out << s.routing << ',' << bounds_name(s.bounds) << ',' << s.weight << ',' << s.metadata_cap << ',' << s.vector_cap << ','
        << o.recall << ',' << o.p50 << ',' << o.p99 << ',' << o.vectors << ',' << o.max_vectors << ',' << o.metadata << ','
        << o.max_metadata << ',' << o.leaves << ',' << o.pruned << ',' << o.skipped << ',' << o.projections << ',' << o.radius << ','
        << o.expansions << ',' << o.positive << ',' << o.improved << ',' << o.routing_us << ',' << o.scanning_us << ',' << o.exhausted << ','
        << o.vectors / double(data.count()) << ',' << index.leaf_count() << ',' << index.raw_bytes() << ','
        << index.overhead_bytes() << ',' << index.graph_bytes() << ',' << index.bounds_bytes(s.bounds) << ',' << build_seconds << ','
        << b.clustering_s << ',' << b.centroids_s << ',' << b.radius_s << ',' << b.projections_s << ',' << b.packing_s << ','
        << b.graph_s << ',' << b.graph_distance_evaluations << ',' << data.truth.size() << ',' << peak_rss << ',' << b.shared_basis_s << ',' << b.shared_ranges_s << ','
        << o.shared_query_projections << ',' << o.shared_intervals << ','
        << (s.scan_order == GraphScanOrder::centroid ? "centroid" : "routing") << ','
        << (index.quantized() ? "int8_sq_per_dim" : "fp32") << ',' << data.count()*data.dimension*sizeof(float) << ',' << b.support_s << '\n';
}
}
