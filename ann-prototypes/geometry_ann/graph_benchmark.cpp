// Build one immutable topology and compare geometry-only navigation changes on identical data.
#include "graph_experiment.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <sys/resource.h>
using namespace geometry_ann;
int main(int argc, char** argv) {
    try {
        auto config = parse_options(argc, argv);
        if (config.variant != "all" && config.variant != "V2") throw std::invalid_argument("graph ablation builds V2 metadata once");
        if (config.ordering) throw std::invalid_argument("use --weights for the graph ordering ablation");
        config.build.variant = Variant::pca;
        if (!config.build.graph_degree) config.build.graph_degree = 8;
        if (!config.build.shared_directions) config.build.shared_directions = 8;
        auto data = config.base.empty() ? synthetic(config.count, config.query_count, config.dimension, config.build.seed)
            : load_dataset(config.base, config.queries, config.truth, config.query_count);
        std::cerr << "Building shared graph: n=" << data.base.size() / data.dimension << " d=" << data.dimension
                  << " queries=" << data.truth.size() << " degree=" << config.build.graph_degree << '\n';
        auto start = std::chrono::steady_clock::now(); Index index(data.base, data.dimension, config.build);
        double build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        rusage usage{}; getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
        std::size_t peak = std::size_t(usage.ru_maxrss);
#else
        std::size_t peak = std::size_t(usage.ru_maxrss) * 1024;
#endif
        const auto& stats = index.build_statistics();
        std::cerr << "Built " << index.leaf_count() << " leaves in " << build_seconds << "s; graph=" << stats.graph_s
                  << "s PCA=" << stats.projections_s << "s; graph bytes=" << index.graph_bytes() << '\n';
        graph_csv_header(std::cout);
        for (const auto& setting : graph_settings(config, index)) {
            auto observation = graph_measure(index, data, config, setting);
            graph_csv_row(std::cout, index, data, setting, observation, build_seconds, peak); std::cout.flush();
            std::cerr << setting.routing << '/' << bounds_name(setting.bounds) << " weight=" << setting.weight
                      << " metadata=" << setting.metadata_cap << " vectors=" << setting.vector_cap
                      << " recall=" << observation.recall << " p50_us=" << observation.p50 << '\n';
        }
    } catch (const std::exception& error) { std::cerr << "error: " << error.what() << '\n'; return 1; }
}
