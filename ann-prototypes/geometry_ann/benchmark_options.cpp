// Validated CLI configuration, including explicit vector budgets and routing ablations.
#include "benchmark_options.h"
#include <algorithm>
#include <map>
#include <cmath>
#include <sstream>
#include <stdexcept>
namespace geometry_ann {
namespace {
std::size_t number(const std::string& text) {
    if (text.empty() || text.front() == '-') throw std::invalid_argument("expected nonnegative integer");
    std::size_t consumed = 0;
    auto value = std::stoull(text, &consumed);
    if (consumed != text.size()) throw std::invalid_argument("invalid integer: " + text);
    return value;
}
}
BenchmarkOptions parse_options(int argc, char** argv) {
    const std::vector<std::string> known = {"--base", "--queries", "--truth", "--n", "--nq", "--dim", "--seed",
        "--coarse", "--children", "--leaves", "--target-size", "--sample", "--iterations", "--directions",
        "--pca-sample", "--metadata", "--variant", "--split", "--vectors", "--leaf-scans", "--routing", "--ordering", "--graph-degree", "--graph-parents", "--graph-refinements", "--graph-seeds", "--metadata-budgets", "--weights", "--bounds", "--controls", "--settings", "--shared-directions", "--shared-sample", "--evaluation-queries", "--evaluation-truth", "--evaluation-output", "--evaluation-nq", "--repeats", "--evaluation-fixed-settings", "--int8-base", "--int8-scales", "--int8-offsets", "--geometry-diagnostics", "--evaluation-group-bounds", "--build-threads", "--search-cpu", "--reassign-parents", "--reassign-iterations", "--memory-output", "--assignments", "--select-by", "--routing-bits", "--routing-candidates", "--routing-scan-bits", "--routing-initial", "--float-routing"};
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::invalid_argument("options require a value; see README.md");
        std::string key = argv[i];
        if (std::find(known.begin(), known.end(), key) == known.end()) throw std::invalid_argument("unknown option " + key);
        args[key] = argv[i + 1];
    }
    auto get = [&](const std::string& key, std::size_t fallback) { return args.contains(key) ? number(args.at(key)) : fallback; };
    BenchmarkOptions options;
    if (args.contains("--evaluation-queries")) options.evaluation_queries = args.at("--evaluation-queries");
    if (args.contains("--evaluation-truth")) options.evaluation_truth = args.at("--evaluation-truth");
    if (args.contains("--evaluation-output")) options.evaluation_output = args.at("--evaluation-output");
    if (args.contains("--evaluation-fixed-settings")) options.evaluation_fixed_settings = args.at("--evaluation-fixed-settings");
    if (args.contains("--int8-base")) options.int8_base = args.at("--int8-base");
    if (args.contains("--int8-scales")) options.int8_scales = args.at("--int8-scales");
    if (args.contains("--int8-offsets")) options.int8_offsets = args.at("--int8-offsets");
    if (!options.int8_offsets.empty() && options.int8_base.empty()) throw std::invalid_argument("SQ8 offsets require codes");
    if (options.int8_base.empty() != options.int8_scales.empty()) throw std::invalid_argument("SQ8 requires codes and scales");
    bool any_evaluation = !options.evaluation_queries.empty() || !options.evaluation_truth.empty() || !options.evaluation_output.empty();
    if (any_evaluation && (options.evaluation_queries.empty() || options.evaluation_truth.empty() || options.evaluation_output.empty()))
        throw std::invalid_argument("evaluation requires queries, truth and output");
    options.evaluation_nq = get("--evaluation-nq", options.evaluation_nq);
    options.repeats = get("--repeats", options.repeats);
    if (!options.evaluation_nq || !options.repeats) throw std::invalid_argument("evaluation counts must be positive");
    auto& build = options.build;
    build.coarse = get("--coarse", build.coarse); build.children = get("--children", build.children);
    build.leaf_budget = get("--leaves", build.coarse * build.children);
    build.target_leaf_size = get("--target-size", build.target_leaf_size);
    build.training_sample = get("--sample", build.training_sample); build.iterations = get("--iterations", build.iterations);
    build.directions = get("--directions", build.directions); build.pca_sample = get("--pca-sample", build.pca_sample);
    build.seed = std::uint32_t(get("--seed", build.seed));
    build.build_threads = get("--build-threads", build.build_threads);
    if (!build.build_threads) throw std::invalid_argument("build threads must be positive");
    if (args.contains("--search-cpu")) options.search_cpu = get("--search-cpu", 0);
    build.reassign_parents = get("--reassign-parents", build.reassign_parents);
    build.reassign_iterations = get("--reassign-iterations", build.reassign_iterations);
    if (args.contains("--memory-output")) options.memory_output = args.at("--memory-output");
    if (args.contains("--assignments")) options.assignments_file = args.at("--assignments");
    build.routing_bits = get("--routing-bits", build.routing_bits);
    if (build.routing_bits > 8) throw std::invalid_argument("routing bits must be 0..8 per dimension");
    build.routing_scan_bits = get("--routing-scan-bits", build.routing_scan_bits);
    if (build.routing_scan_bits > 8) throw std::invalid_argument("routing scan bits must be 0..8");
    options.routing_candidates = get("--routing-candidates", options.routing_candidates);
    if (!options.routing_candidates) throw std::invalid_argument("routing candidates must be positive");
    options.routing_initial = get("--routing-initial", options.routing_initial);
    auto float_routing = get("--float-routing", 0);
    if (float_routing > 1) throw std::invalid_argument("float-routing must be 0 or 1");
    options.float_routing = build.float_graph_build = float_routing == 1;
    if (args.contains("--select-by")) {
        const auto& criterion = args.at("--select-by");
        if (criterion != "p50" && criterion != "vectors") throw std::invalid_argument("select-by must be p50 or vectors");
        options.select_by_vectors = criterion == "vectors";
    }
    if (args.contains("--split")) options.split = args.at("--split");
    if (options.split != "variance" && options.split != "kmeans") throw std::invalid_argument("split must be variance or kmeans");
    build.local_kmeans_split = options.split == "kmeans";
    if (args.contains("--variant")) options.variant = args.at("--variant");
    if (options.variant != "all" && options.variant != "V0" && options.variant != "V1" && options.variant != "V2")
        throw std::invalid_argument("variant must be all, V0, V1, or V2");
    options.count = get("--n", options.count); options.query_count = get("--nq", options.query_count);
    options.dimension = get("--dim", options.dimension);
    if (!options.query_count) throw std::invalid_argument("nq must be positive");
    if (args.contains("--base")) {
        if (!args.contains("--queries") || !args.contains("--truth")) throw std::invalid_argument("base requires queries and truth");
        options.base = args.at("--base"); options.queries = args.at("--queries"); options.truth = args.at("--truth");
    } else {
        if (args.contains("--queries") || args.contains("--truth")) throw std::invalid_argument("queries/truth require base");
        if (options.count < 10) throw std::invalid_argument("recall@10 requires n >= 10");
    }
    if (args.contains("--metadata")) options.metadata_cap = get("--metadata", 0);
    if (args.contains("--leaf-scans")) options.leaf_scans = get("--leaf-scans", 0);
    if (args.contains("--routing")) {
        auto routing = args.at("--routing");
        if (routing != "hierarchy" && routing != "flat" && routing != "both")
            throw std::invalid_argument("routing must be hierarchy, flat, or both");
        options.routing = routing == "flat" ? RoutingMode::flat : routing == "both" ? RoutingMode::both : RoutingMode::hierarchy;
    }
    if (args.contains("--ordering") && args.at("--ordering") != "all") {
        const auto& order = args.at("--ordering");
        if (order != "centroid" && order != "bound" && order != "variance")
            throw std::invalid_argument("ordering must be all, centroid, bound, or variance");
        options.ordering = order == "centroid" ? Ordering::centroid : order == "bound" ? Ordering::bound : Ordering::variance;
        if (options.variant == "V0" && *options.ordering != Ordering::centroid)
            throw std::invalid_argument("V0 supports centroid ordering only");
    }
    if (args.contains("--vectors")) {
        options.vector_caps.clear();
        std::istringstream input(args.at("--vectors")); std::string token;
        while (std::getline(input, token, ',')) options.vector_caps.push_back(number(token));
        if (options.vector_caps.empty()) throw std::invalid_argument("vectors needs at least one integer cap");
    }
    auto group_bounds=get("--evaluation-group-bounds",0);
    if (group_bounds>1) throw std::invalid_argument("evaluation-group-bounds must be 0 or 1");
    options.evaluation_group_bounds=group_bounds;
    if (args.contains("--geometry-diagnostics")) options.geometry_diagnostics = args.at("--geometry-diagnostics");
    build.shared_directions = get("--shared-directions", build.shared_directions);
    build.shared_sample = get("--shared-sample", build.shared_sample);
    build.graph_degree = get("--graph-degree", build.graph_degree); build.graph_parent_probes = get("--graph-parents", build.graph_parent_probes);
    build.graph_refinements = get("--graph-refinements", build.graph_refinements); options.graph_seeds = get("--graph-seeds", options.graph_seeds);
    options.graph_controls = get("--controls", 1) != 0;
    if (args.contains("--settings")) options.settings_file = args.at("--settings");
    auto parse_list = [&](const std::string& key, auto& values, auto parse) {
        if (!args.contains(key)) return;
        values.clear(); std::istringstream input(args.at(key)); std::string token;
        while (std::getline(input, token, ',')) values.push_back(parse(token));
        if (values.empty()) throw std::invalid_argument(key + " must not be empty");
    };
    parse_list("--metadata-budgets", options.graph_metadata_budgets, number);
    parse_list("--weights", options.geometry_weights, [](const std::string& token) {
        std::size_t consumed = 0; double value = std::stod(token, &consumed);
        if (consumed != token.size() || !std::isfinite(value) || value < 0 || value > 1)
            throw std::invalid_argument("weights must be within [0,1]");
        return value;
    });
    if (args.contains("--bounds")) {
        const auto& mode = args.at("--bounds");
        if (mode == "none") options.selected_bounds = BoundsMode::none;
        else if (mode == "radius") options.selected_bounds = BoundsMode::radius;
        else if (mode == "pca-max") options.selected_bounds = BoundsMode::pca_max;
        else if (mode == "support") options.selected_bounds = BoundsMode::support;
        else if (mode == "shared-residual") options.selected_bounds = BoundsMode::shared_residual;
        else if (mode == "shared") options.selected_bounds = BoundsMode::shared;
        else if (mode == "pca") options.selected_bounds = BoundsMode::pca;
        else if (mode != "all") throw std::invalid_argument("bounds must be all, none, radius, pca-max, or pca");
    }
    return options;
}
}
