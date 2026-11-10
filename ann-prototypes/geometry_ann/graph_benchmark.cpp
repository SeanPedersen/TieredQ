// Build one immutable topology and compare geometry-only navigation changes on identical data.
#include "graph_experiment.h"
#include <chrono>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <sys/resource.h>
using namespace geometry_ann;
int main(int argc, char** argv) {
    try {
        auto config = parse_options(argc, argv);
        if (config.variant != "all" && config.variant != "V2" && config.variant != "V0")
            throw std::invalid_argument("graph ablation supports V0 partitions or V2 adaptive partitions");
        if (config.ordering) throw std::invalid_argument("use --weights for the graph ordering ablation");
        config.build = graph_build_options(config);
        auto data = config.base.empty() ? synthetic(config.count, config.query_count, config.dimension, config.build.seed)
            : load_dataset(config.base, config.queries, config.truth, config.query_count);
        auto quantized = config.int8_base.empty() ? QuantizedDataset{} : load_quantized(config.int8_base,config.int8_scales,data.count(),data.dimension,config.int8_offsets);
        std::cerr << "Storage=" << (quantized.codes.empty() ? "FP32" : "SQ8") << '\n';
        std::cerr << "Building shared graph: n=" << data.count() << " d=" << data.dimension
                  << " queries=" << data.truth.size() << " degree=" << config.build.graph_degree << '\n';
        auto start = std::chrono::steady_clock::now(); Index index(data.base, data.dimension, config.build, quantized.view());
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
        if (!config.evaluation_output.empty()) { data.release_base(); quantized.release_codes(); }
        std::vector<std::pair<GraphSetting, GraphObservation>> calibration;
        graph_csv_header(std::cout);
        for (const auto& setting : graph_settings(config, index)) {
            auto observation = graph_measure(index, data, config, setting);
            calibration.push_back({setting, observation});
            graph_csv_row(std::cout, index, data, setting, observation, build_seconds, peak); std::cout.flush();
            std::cerr << setting.routing << '/' << bounds_name(setting.bounds) << " weight=" << setting.weight
                      << " metadata=" << setting.metadata_cap << " vectors=" << setting.vector_cap
                      << " recall=" << observation.recall << " p50_us=" << observation.p50 << '\n';
        }
        if (!config.evaluation_output.empty()) graph_evaluate(index, config, calibration, build_seconds, peak);
        if (!config.geometry_diagnostics.empty()) {
            if (std::filesystem::exists(config.geometry_diagnostics)) throw std::runtime_error("diagnostics output exists");
            std::ofstream out(config.geometry_diagnostics);
            if (!out) throw std::runtime_error("cannot write diagnostics");
            out << "query,bounds,weight,positive_bounds,bounds_above_reference,unsafe_members,mean_truth_cluster_rank,max_truth_cluster_rank,mean_bound_ratio,truth_clusters,oracle_vectors_9,oracle_vectors_10,ranked_vectors_9,ranked_vectors_10,witness_ordering,witness_evaluations\n";
            auto count=std::min<std::size_t>(50,data.truth.size());
            for (const auto& d:index.diagnose_geometry(std::span(data.queries).first(count*data.dimension),std::span(data.truth).first(count)))
                out << d.query << ',' << bounds_name(d.mode) << ',' << d.weight << ',' << d.positive << ','
                    << d.above_reference << ',' << d.unsafe_members << ',' << d.mean_truth_cluster_rank << ','
                    << d.max_truth_cluster_rank << ',' << d.mean_bound_ratio << ',' << d.truth_clusters << ','
                    << d.oracle_vectors_9 << ',' << d.oracle_vectors_10 << ',' << d.ranked_vectors_9 << ',' << d.ranked_vectors_10 << ',' << d.witness_ordering << ',' << d.witness_evaluations << '\n';
        }
    } catch (const std::exception& error) { std::cerr << "error: " << error.what() << '\n'; return 1; }
}
