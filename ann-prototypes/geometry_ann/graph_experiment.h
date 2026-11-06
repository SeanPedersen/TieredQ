// Frozen graph experiment settings and query measurements, independent of index internals.
#pragma once
#include "benchmark_options.h"
#include "dataset.h"
#include <iosfwd>
namespace geometry_ann {
struct GraphSetting {
    std::string routing;
    BoundsMode bounds;
    double weight;
    std::size_t metadata_cap, vector_cap;
    GraphScanOrder scan_order = GraphScanOrder::centroid;
};
struct GraphObservation {
    double recall = 0, p50 = 0, p99 = 0, vectors = 0, metadata = 0, leaves = 0, pruned = 0, skipped = 0;
    double projections = 0, radius = 0, expansions = 0, positive = 0, improved = 0, routing_us = 0, scanning_us = 0, exhausted = 0;
    double shared_query_projections = 0, shared_intervals = 0;
    std::size_t max_vectors = 0, max_metadata = 0;
};
const char* bounds_name(BoundsMode mode);
std::vector<GraphSetting> graph_settings(const BenchmarkOptions& options, const Index& index);
GraphObservation graph_measure(const Index& index, const Dataset& data, const BenchmarkOptions& config, const GraphSetting& setting);
void graph_csv_header(std::ostream& output);
void graph_csv_row(std::ostream& output, const Index& index, const Dataset& data, const GraphSetting& setting,
                   const GraphObservation& observation, double build_seconds, std::size_t peak_rss);
}
