// Measured search sweeps and matched-recall selections, with scan fraction as a primary gate.
#include "benchmark_options.h"
#include "dataset.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <sys/resource.h>
using namespace geometry_ann;
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
std::size_t peak_rss() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage)) return 0;
#ifdef __APPLE__
    return std::size_t(usage.ru_maxrss);
#else
    return std::size_t(usage.ru_maxrss) * 1024;
#endif
}
struct Observation {
    double recall, p50, p99, vectors, metadata, pruned, exhausted, skipped;
    std::size_t budget;
};
struct BuildMeasurement { double seconds; std::size_t peak; };
Observation measure(const Index& index, const Dataset& data, const SearchOptions& options) {
    index.search(std::span(data.queries).first(data.dimension), options);
    std::vector<double> latency;
    double hits = 0, vectors = 0, metadata = 0, pruned = 0, exhausted = 0, skipped = 0;
    for (std::size_t q = 0; q < data.truth.size(); ++q) {
        auto query = std::span(data.queries).subspan(q * data.dimension, data.dimension);
        auto start = Clock::now(); auto result = index.search(query, options);
        latency.push_back(seconds(start) * 1e6);
        for (const auto& neighbor : result.neighbors)
            hits += std::find(data.truth[q].begin(), data.truth[q].end(), neighbor.id) != data.truth[q].end();
        vectors += result.vectors_scanned; metadata += result.metadata_evaluations;
        pruned += result.pruned; exhausted += result.metadata_exhausted; skipped += result.leaves_skipped_for_vector_budget;
    }
    std::sort(latency.begin(), latency.end());
    auto percentile = [&](double p) { return latency[std::size_t(std::ceil(p * latency.size())) - 1]; };
    double queries = double(data.truth.size());
    return {hits / (10 * queries), percentile(.50), percentile(.99), vectors / queries, metadata / queries,
            pruned / queries, exhausted / queries, skipped / queries, options.leaf_scans};
}
void report_targets(const std::vector<Observation>& observations, const std::string& label, std::size_t cap) {
    for (double target : {.95, .99}) {
        const Observation* best = nullptr;
        for (const auto& observation : observations)
            if (observation.recall >= target && (!best || observation.p50 < best->p50)) best = &observation;
        std::cerr << label << " vector_cap=" << cap << " target=" << target;
        if (best) std::cerr << " leaves=" << best->budget << " recall=" << best->recall << " p50_us=" << best->p50;
        else std::cerr << " NOT_REACHED";
        std::cerr << '\n';
    }
}
void run_sweeps(const Index& index, const Dataset& data, const BenchmarkOptions& config,
                const std::string& name, BuildMeasurement build, bool flat_routing) {
    std::vector<std::size_t> budgets;
    if (config.leaf_scans) budgets.push_back(*config.leaf_scans);
    else {
        for (std::size_t budget = 1; budget < index.leaf_count(); budget *= 2) budgets.push_back(budget);
        budgets.push_back(index.leaf_count());
    }
    std::string routing = flat_routing ? "flat" : "hierarchy";
    double count = double(data.count());
    for (auto ordering : {Ordering::centroid, Ordering::bound, Ordering::variance}) {
        if (name == "V0" && ordering != Ordering::centroid) continue;
        if (config.ordering && ordering != *config.ordering) continue;
        std::string order = ordering == Ordering::centroid ? "centroid" : ordering == Ordering::bound ? "bound" : "variance";
        for (auto cap : config.vector_caps) {
            std::vector<Observation> observations;
            for (auto budget : budgets) {
                SearchOptions search; search.leaf_scans = budget; search.ordering = ordering;
                search.metadata_evaluations = config.metadata_cap.value_or(index.leaf_count() + config.build.coarse);
                search.vector_scans = cap; search.routing = flat_routing ? Routing::flat : Routing::hierarchy;
                auto observation = measure(index, data, search); observations.push_back(observation);
                const auto& o = observation;
                std::cerr << name << '/' << order << '/' << routing << " leaves=" << budget << " recall=" << o.recall
                          << " scan_fraction=" << o.vectors / count << " p50_us=" << o.p50 << '\n';
                std::cout << name << ',' << order << ',' << budget << ',' << search.metadata_evaluations << ','
                    << index.leaf_count() << ',' << build.seconds << ',' << index.raw_bytes() << ',' << index.overhead_bytes()
                    << ',' << build.peak << ',' << o.recall << ',' << o.p50 << ',' << o.p99 << ',' << o.vectors << ','
                    << o.metadata << ',' << o.pruned << ',' << o.exhausted << ',' << routing << ',' << cap << ','
                    << o.vectors / count << ',' << o.skipped << '\n';
            }
            report_targets(observations, name + '/' + order + '/' + routing, cap);
        }
    }
}
}
int main(int argc, char** argv) {
    try {
        auto config = parse_options(argc, argv);
        if (!config.int8_base.empty() || !config.evaluation_output.empty()) throw std::invalid_argument("SQ8 and held-out replay require graph_bench");
        auto data = config.base.empty() ? synthetic(config.count, config.query_count, config.dimension, config.build.seed)
            : load_dataset(config.base, config.queries, config.truth, config.query_count);
        std::cerr << "dataset n=" << data.count() << " nq=" << data.truth.size() << " d=" << data.dimension
                  << " seed=" << config.build.seed << " split=" << config.split << '\n';
        std::cout << "variant,ordering,leaf_budget,metadata_cap,leaves,build_s,raw_bytes,overhead_bytes,process_peak_rss_bytes,recall10,p50_us,p99_us,mean_vectors,mean_metadata,mean_pruned,metadata_exhaustion_rate,routing,vector_cap,mean_scan_fraction,mean_skipped_leaves\n";
        const std::vector<std::pair<Variant, std::string>> variants = {{Variant::fixed,"V0"},{Variant::radius,"V1"},{Variant::pca,"V2"}};
        for (const auto& [variant, name] : variants) {
            if (config.variant != "all" && config.variant != name) continue;
            config.build.variant = variant;
            auto start = Clock::now(); Index index(data.base, data.dimension, config.build);
            BuildMeasurement build{seconds(start), peak_rss()};
            std::cerr << name << " built in " << build.seconds << "s; " << index.leaf_count() << " leaves\n";
            if (config.routing != RoutingMode::flat) run_sweeps(index, data, config, name, build, false);
            if (config.routing != RoutingMode::hierarchy) run_sweeps(index, data, config, name, build, true);
        }
    } catch (const std::exception& error) { std::cerr << "error: " << error.what() << '\n'; return 1; }
}
