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
        "--pca-sample", "--metadata", "--variant", "--split", "--vectors", "--leaf-scans", "--routing", "--ordering", "--graph-degree", "--graph-parents", "--graph-refinements", "--graph-seeds", "--metadata-budgets", "--weights", "--bounds", "--controls", "--settings", "--shared-directions", "--shared-sample"};
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::invalid_argument("options require a value; see README.md");
        std::string key = argv[i];
        if (std::find(known.begin(), known.end(), key) == known.end()) throw std::invalid_argument("unknown option " + key);
        args[key] = argv[i + 1];
    }
    auto get = [&](const std::string& key, std::size_t fallback) { return args.contains(key) ? number(args.at(key)) : fallback; };
    BenchmarkOptions options;
    auto& build = options.build;
    build.coarse = get("--coarse", build.coarse); build.children = get("--children", build.children);
    build.leaf_budget = get("--leaves", build.coarse * build.children);
    build.target_leaf_size = get("--target-size", build.target_leaf_size);
    build.training_sample = get("--sample", build.training_sample); build.iterations = get("--iterations", build.iterations);
    build.directions = get("--directions", build.directions); build.pca_sample = get("--pca-sample", build.pca_sample);
    build.seed = std::uint32_t(get("--seed", build.seed));
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
        else if (mode == "shared") options.selected_bounds = BoundsMode::shared;
        else if (mode == "pca") options.selected_bounds = BoundsMode::pca;
        else if (mode != "all") throw std::invalid_argument("bounds must be all, none, radius, pca-max, or pca");
    }
    return options;
}
}
