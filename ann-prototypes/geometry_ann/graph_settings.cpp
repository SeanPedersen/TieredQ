// Factorial ablation settings and replay of calibration-selected configurations.
#include "graph_experiment.h"
#include <fstream>
#include <algorithm>
#include <sstream>
#include <stdexcept>
namespace geometry_ann {
const char* bounds_name(BoundsMode mode) {
    switch (mode) {
        case BoundsMode::none: return "none";
        case BoundsMode::radius: return "radius";
        case BoundsMode::pca_max: return "pca-max";
        case BoundsMode::pca: return "pca";
        case BoundsMode::shared: return "shared";
        default: return "automatic";
    }
}
namespace {
BoundsMode parse_mode(const std::string& text) {
    for (auto mode : {BoundsMode::none, BoundsMode::radius, BoundsMode::pca_max, BoundsMode::pca, BoundsMode::shared})
        if (text == bounds_name(mode)) return mode;
    throw std::invalid_argument("invalid bounds mode in settings file");
}
std::vector<GraphSetting> load_settings(const std::string& filename) {
    std::ifstream input(filename);
    if (!input) throw std::runtime_error("cannot open settings file");
    std::string line; std::getline(input, line);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    bool scan_column = line == "routing,bounds,weight,metadata_cap,vector_cap,scan_order";
    if (!scan_column && line != "routing,bounds,weight,metadata_cap,vector_cap") throw std::invalid_argument("invalid settings header");
    std::vector<GraphSetting> result;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream record(line); GraphSetting setting; setting.scan_order = GraphScanOrder::routing; std::string mode, extra;
        if (!(record >> setting.routing >> mode >> setting.weight >> setting.metadata_cap >> setting.vector_cap))
            throw std::invalid_argument("invalid settings record");
        if (scan_column) {
            std::string scan;
            if (!(record >> scan) || (scan != "centroid" && scan != "routing"))
                throw std::invalid_argument("invalid graph scan order");
            setting.scan_order = scan == "centroid" ? GraphScanOrder::centroid : GraphScanOrder::routing;
        }
        if (record >> extra) throw std::invalid_argument("extra fields in settings record");
        if (setting.routing != "graph" && setting.routing != "flat" && setting.routing != "hierarchy")
            throw std::invalid_argument("invalid routing in settings file");
        setting.bounds = parse_mode(mode); result.push_back(setting);
    }
    if (result.empty()) throw std::invalid_argument("empty settings file");
    return result;
}
}
std::vector<GraphSetting> graph_settings(const BenchmarkOptions& config, const Index& index) {
    if (!config.settings_file.empty()) return load_settings(config.settings_file);
    std::vector<GraphSetting> result;
    for (auto mode : {BoundsMode::none, BoundsMode::radius, BoundsMode::pca_max, BoundsMode::pca, BoundsMode::shared}) {
        if (config.selected_bounds != BoundsMode::automatic && config.selected_bounds != mode) continue;
        auto weights = mode == BoundsMode::none ? std::vector<double>{0} : config.geometry_weights;
        for (double weight : weights) for (auto vectors : config.vector_caps) {
            auto budgets = config.metadata_cap ? std::vector<std::size_t>{*config.metadata_cap} : config.graph_metadata_budgets;
            for (auto budget : budgets) result.push_back({"graph", mode, weight, budget, vectors});
            if (config.graph_controls) {
                auto full = index.leaf_count() + config.build.coarse;
                result.push_back({"flat", mode, weight, full, vectors, GraphScanOrder::routing});
                result.push_back({"hierarchy", mode, weight, full, vectors, GraphScanOrder::routing});
            }
        }
    }
    return result;
}
}
