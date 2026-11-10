// Benchmark configuration parsing, kept separate from index execution and measurement.
#pragma once
#include "index.h"
#include <optional>
#include <string>
namespace geometry_ann {
enum class RoutingMode { hierarchy, flat, both };
struct BenchmarkOptions {
    BuildOptions build;
    std::size_t count = 10000, query_count = 100, dimension = 32;
    std::optional<std::size_t> leaf_scans, metadata_cap;
    std::optional<Ordering> ordering;
    std::vector<std::size_t> graph_metadata_budgets = {256, 512, 1024, 2048};
    std::vector<double> geometry_weights = {0, 0.5, 1};
    BoundsMode selected_bounds = BoundsMode::automatic;
    std::size_t graph_seeds = 4, routing_candidates = 4096, routing_initial = 0;
    bool graph_controls = true, evaluation_group_bounds = false;
    bool select_by_vectors = false, float_routing = false;  // calibration picks minimise mean scanned vectors instead of p50 latency
    std::string geometry_diagnostics, settings_file, int8_base, int8_scales, int8_offsets, memory_output, assignments_file;
    std::string evaluation_queries, evaluation_truth, evaluation_output, evaluation_fixed_settings;
    std::size_t evaluation_nq = 750, repeats = 3;
    std::optional<std::size_t> search_cpu;  // pin after a multi-threaded build so search stays single-core
    std::vector<std::size_t> vector_caps = {std::numeric_limits<std::size_t>::max()};
    RoutingMode routing = RoutingMode::hierarchy;
    std::string base, queries, truth, variant = "all", split = "variance";
};
BenchmarkOptions parse_options(int argc, char** argv);
}
