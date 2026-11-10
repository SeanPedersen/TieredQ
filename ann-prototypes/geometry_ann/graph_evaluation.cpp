// Freeze calibration choices before loading held-out queries; reuse one built index.
#include "graph_experiment.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <sys/resource.h>
namespace geometry_ann {
namespace {
std::ofstream exclusive(const std::string& name) {
    if (std::filesystem::exists(name)) throw std::runtime_error("evaluation output already exists: " + name);
    std::ofstream result(name); if (!result) throw std::runtime_error("cannot write " + name); return result;
}
std::pair<std::size_t,std::size_t> memory() {
    std::ifstream status("/proc/self/status"); std::string line; std::size_t rss = 0, swap = 0;
    while (std::getline(status, line)) {
        std::istringstream value(line); std::string key; std::size_t kb = 0;
        if (value >> key >> kb) { if (key == "VmRSS:") rss = kb * 1024; if (key == "VmSwap:") swap = kb * 1024; }
    }
    return {rss, swap};
}
}
void graph_evaluate(const Index& index, const BenchmarkOptions& config,
                    const std::vector<std::pair<GraphSetting, GraphObservation>>& calibration,
                    double build_seconds, std::size_t peak_rss) {
    auto selection = exclusive(config.evaluation_output + ".selection.csv");
    selection << "target,routing,status,bounds,weight,metadata_cap,vector_cap,scan_order,calibration_recall,calibration_p50_us\n";
    std::vector<GraphSetting> selected;
    std::set<std::tuple<std::string,BoundsMode,double,std::size_t,std::size_t,GraphScanOrder>> keys;
    auto groups=config.evaluation_group_bounds
        ? std::vector<BoundsMode>{BoundsMode::none,BoundsMode::radius,BoundsMode::pca_max,BoundsMode::pca,BoundsMode::shared,BoundsMode::shared_residual,BoundsMode::support}
        : std::vector<BoundsMode>{BoundsMode::automatic};
    for (double target : {.95,.99,.995}) for (const std::string route : {"graph", "flat", "hierarchy"}) for (auto group:groups) {
        const std::pair<GraphSetting,GraphObservation>* best = nullptr;
        for (const auto& candidate : calibration)
            if (candidate.first.routing == route && (!config.evaluation_group_bounds || candidate.first.bounds==group) && candidate.second.recall >= target &&
                (!best || candidate.second.p50 < best->second.p50)) best = &candidate;
        bool reached = best != nullptr;
        if (!best) for (const auto& candidate : calibration)
            if (candidate.first.routing == route && (!config.evaluation_group_bounds || candidate.first.bounds==group) && (!best || candidate.second.recall > best->second.recall ||
                (candidate.second.recall == best->second.recall && candidate.second.p50 < best->second.p50))) best = &candidate;
        if (!best) continue;
        const auto& s = best->first;
        selection << target << ',' << route << (reached ? ",SELECTED," : ",NOT_REACHED_DIAGNOSTIC,")
                  << bounds_name(s.bounds) << ',' << s.weight << ',' << s.metadata_cap << ',' << s.vector_cap << ','
                  << (s.scan_order == GraphScanOrder::centroid ? "centroid" : "routing") << ','
                  << best->second.recall << ',' << best->second.p50 << '\n';
        if (keys.emplace(s.routing,s.bounds,s.weight,s.metadata_cap,s.vector_cap,s.scan_order).second) selected.push_back(s);
    }
    if (!config.evaluation_fixed_settings.empty()) {
        auto replay = config; replay.settings_file = config.evaluation_fixed_settings;
        for (const auto& s : graph_settings(replay,index)) {
            selection << "SIFT_REPLAY," << s.routing << ",FIXED," << bounds_name(s.bounds) << ',' << s.weight << ','
                      << s.metadata_cap << ',' << s.vector_cap << ','
                      << (s.scan_order == GraphScanOrder::centroid ? "centroid" : "routing") << ",,\n";
            if (keys.emplace(s.routing,s.bounds,s.weight,s.metadata_cap,s.vector_cap,s.scan_order).second) selected.push_back(s);
        }
    }
    selection.close(); // Choices are persisted before any held-out data is read.
    auto data = load_dataset(config.base, config.evaluation_queries, config.evaluation_truth, config.evaluation_nq);
    data.release_base();
    auto out = exclusive(config.evaluation_output), mem = exclusive(config.evaluation_output + ".memory.csv");
    out << "repeat,"; graph_csv_header(out);
    mem << "repeat,routing,metadata_cap,vector_cap,rss_bytes_before,rss_bytes_after,swap_bytes_before,swap_bytes_after,major_faults\n";
    for (std::size_t repeat = 0; repeat < config.repeats; ++repeat) for (const auto& setting : selected) {
        auto before = memory(); rusage start{}, end{}; getrusage(RUSAGE_SELF, &start);
        auto observation = graph_measure(index, data, config, setting);
        getrusage(RUSAGE_SELF, &end); auto after = memory();
        out << repeat << ','; graph_csv_row(out,index,data,setting,observation,build_seconds,peak_rss); out.flush();
        mem << repeat << ',' << setting.routing << ',' << setting.metadata_cap << ',' << setting.vector_cap << ','
            << before.first << ',' << after.first << ',' << before.second << ',' << after.second << ',' << end.ru_majflt - start.ru_majflt << '\n'; mem.flush();
    }
}
}
