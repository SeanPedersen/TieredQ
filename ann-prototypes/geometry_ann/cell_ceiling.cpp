// Cell-ceiling study driver: build fixed hierarchical k-means leaves, then measure how many
// leaves each convex lower bound certifies against the 10th true-neighbor distance.
#include "benchmark_options.h"
#include "dataset.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <thread>
using namespace geometry_ann;
namespace {
struct CellCli {
    CellDiagnosticOptions diagnostic;
    std::size_t queries = 50;
    std::string output;
};
std::size_t parse_count(const std::string& text) {
    std::size_t consumed = 0; auto value = std::stoull(text, &consumed);
    if (consumed != text.size() || text.front() == '-') throw std::invalid_argument("invalid integer: " + text);
    return value;
}
// Consumes cell-specific flags; everything else is forwarded to the shared benchmark parser.
CellCli split_arguments(int argc, char** argv, std::vector<char*>& forwarded) {
    CellCli cli; cli.diagnostic.threads = std::max(1u, std::thread::hardware_concurrency());
    std::map<std::string, std::size_t*> numeric = {
        {"--cell-directions", &cli.diagnostic.directions}, {"--cell-small-directions", &cli.diagnostic.small_directions},
        {"--hull-iterations", &cli.diagnostic.hull_iterations}, {"--threads", &cli.diagnostic.threads},
        {"--cell-queries", &cli.queries}, {"--cheap-sweeps", &cli.diagnostic.cheap_sweeps},
        {"--converged-sweeps", &cli.diagnostic.converged_sweeps}, {"--rank-iterations", &cli.diagnostic.rank_iterations}};
    forwarded.push_back(argv[0]);
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::invalid_argument("options require a value");
        std::string key = argv[i], value = argv[i + 1];
        if (auto found = numeric.find(key); found != numeric.end()) *found->second = parse_count(value);
        else if (key == "--cell-output") cli.output = value;
        else if (key == "--hull") cli.diagnostic.hull = parse_count(value) != 0;
        else if (key == "--hull-rank") cli.diagnostic.hull_rank = parse_count(value) != 0;
        else { forwarded.push_back(argv[i]); forwarded.push_back(argv[i + 1]); }
    }
    if (cli.output.empty()) throw std::invalid_argument("--cell-output is required");
    if (!cli.queries || !cli.diagnostic.threads) throw std::invalid_argument("cell queries and threads must be positive");
    return cli;
}
void write_csv(std::ostream& out, const CellStudy& study) {
    out << "query,reference,truth_clusters,oracle_vectors_10,score,certified_clusters,certified_vectors,"
           "undecided_clusters,undecided_vectors,pure_vectors_10,hybrid_vectors_10,mean_truth_rank,"
           "mean_truth_bound_ratio,unsafe,bound_ns_per_leaf,hull_iterations,ranked\n";
    out.precision(17);
    for (const auto& row : study.queries) for (const auto& s : row.scores)
        out << row.query << ',' << row.reference << ',' << row.truth_clusters << ',' << row.oracle_vectors_10 << ','
            << s.name << ',' << s.certified_clusters << ',' << s.certified_vectors << ',' << s.undecided_clusters << ','
            << s.undecided_vectors << ',' << s.pure_vectors_10 << ',' << s.hybrid_vectors_10 << ',' << s.mean_truth_rank << ','
            << s.mean_truth_bound_ratio << ',' << s.unsafe << ',' << s.bound_ns_per_leaf << ',' << s.hull_iterations << ','
            << s.ranked << '\n';
}
}
int main(int argc, char** argv) {
    try {
        std::vector<char*> forwarded;
        auto cli = split_arguments(argc, argv, forwarded);
        auto config = parse_options(int(forwarded.size()), forwarded.data());
        if (std::filesystem::exists(cli.output)) throw std::runtime_error("cell output exists: " + cli.output);
        auto build = config.build;
        build.variant = Variant::fixed; build.geometry_metadata = true; build.graph_degree = 0; build.directions = 0;
        auto data = config.base.empty() ? synthetic(config.count, config.query_count, config.dimension, build.seed)
            : load_dataset(config.base, config.queries, config.truth, config.query_count);
        auto quantized = config.int8_base.empty() ? QuantizedDataset{}
            : load_quantized(config.int8_base, config.int8_scales, data.count(), data.dimension, config.int8_offsets);
        std::cerr << "Building fixed leaves: n=" << data.count() << " d=" << data.dimension
                  << " storage=" << (quantized.codes.empty() ? "FP32" : "SQ8") << '\n';
        auto start = std::chrono::steady_clock::now();
        Index index(data.base, data.dimension, build, quantized.view());
        double build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        data.release_base(); quantized.release_codes();
        auto count = std::min(cli.queries, data.truth.size());
        auto study = index.diagnose_cells(std::span(data.queries).first(count * data.dimension),
                                          std::span(data.truth).first(count), cli.diagnostic);
        std::ofstream out(cli.output);
        if (!out) throw std::runtime_error("cannot write " + cli.output);
        write_csv(out, study);
        std::ofstream meta(cli.output + ".meta.json");
        meta << "{\"leaves\": " << index.leaf_count() << ", \"build_s\": " << build_s
             << ", \"slab_build_s\": " << study.slab_build_s << ", \"slab_bytes\": " << study.slab_bytes
             << ", \"directions\": " << cli.diagnostic.directions << ", \"small_directions\": " << cli.diagnostic.small_directions
             << ", \"hull_iterations\": " << cli.diagnostic.hull_iterations
             << ", \"cheap_sweeps\": " << cli.diagnostic.cheap_sweeps << ", \"converged_sweeps\": " << cli.diagnostic.converged_sweeps << ", \"threads\": " << cli.diagnostic.threads
             << ", \"queries\": " << count << "}\n";
        std::cerr << "Leaves=" << index.leaf_count() << " build_s=" << build_s << " slab_build_s=" << study.slab_build_s
                  << " slab_bytes=" << study.slab_bytes << '\n';
    } catch (const std::exception& error) { std::cerr << "error: " << error.what() << '\n'; return 1; }
}
