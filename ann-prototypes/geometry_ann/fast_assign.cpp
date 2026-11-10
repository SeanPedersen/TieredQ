// Assigns every SQ8 row to its nearest centroid among the members of its top-P super-centroids and measures exactness on a sample
// against brute force (labels as in train_partitions.py; one uint32 per row).
#include "dataset.h"
#include "restricted_assignment.h"
#include "internal.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <random>
#include <stdexcept>
#include <string>
using namespace geometry_ann;
namespace {
std::map<std::string, std::string> arguments(int argc, char** argv) {
    std::map<std::string, std::string> result;
    for (int i = 1; i + 1 < argc; i += 2) result[argv[i]] = argv[i + 1];
    if (argc % 2 == 0) throw std::invalid_argument("options require values");
    return result;
}
std::vector<float> read_floats(const std::string& path, std::size_t count) {
    std::ifstream input(path, std::ios::binary); std::vector<float> values(count);
    if (!input.read(reinterpret_cast<char*>(values.data()), std::streamsize(count * sizeof(float))) || input.peek() != EOF)
        throw std::runtime_error("centroid file size mismatch: " + path);
    return values;
}
}
int main(int argc, char** argv) {
    try {
        auto args = arguments(argc, argv);
        auto need = [&](const std::string& key) { if (!args.contains(key)) throw std::invalid_argument("missing " + key); return args.at(key); };
        auto number = [&](const std::string& key, std::size_t fallback) { return args.contains(key) ? std::stoull(args.at(key)) : fallback; };
        const std::size_t rows = std::stoull(need("--rows")), dimension = std::stoull(need("--dim")), centroids_count = std::stoull(need("--k"));
        AssignmentOptions options;
        options.supers = number("--supers", options.supers); options.probes = number("--probes", options.probes);
        options.threads = number("--threads", options.threads);
        const std::size_t check_rows = number("--check-rows", 20000);
        auto data = load_quantized(need("--codes"), need("--scales"), rows, dimension, need("--offsets"));
        auto centroids = read_floats(need("--centroids"), centroids_count * dimension);
        auto decode = [&](std::size_t first, std::size_t count, std::span<float> out) {
            for (std::size_t r = 0; r < count; ++r) for (std::size_t j = 0; j < dimension; ++j)
                out[r * dimension + j] = float(data.codes[(first + r) * dimension + j]) * data.scales[j] + data.offsets[j];
        };
        std::vector<std::uint32_t> labels(rows);
        auto start = std::chrono::steady_clock::now();
        auto stats = restricted_assign(rows, centroids, dimension, options, decode, labels);
        double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        // Exactness: compare against brute force on a seeded sample, by label and by distance (ties and rounding).
        std::vector<std::size_t> sample(std::min(check_rows, rows));
        std::mt19937_64 rng(options.seed); std::uniform_int_distribution<std::size_t> pick(0, rows - 1);
        for (auto& row : sample) row = pick(rng);
        std::vector<float> block(sample.size() * dimension);
        for (std::size_t i = 0; i < sample.size(); ++i) decode(sample[i], 1, std::span(block).subspan(i * dimension, dimension));
        start = std::chrono::steady_clock::now();
        auto exact = brute_force_assign(block, centroids, dimension, options.threads);
        double brute_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::size_t same = 0, worse = 0; double excess = 0, worst = 0, walk_objective = 0, exact_objective = 0;
        for (std::size_t i = 0; i < sample.size(); ++i) {
            auto row = std::span<const float>(block).subspan(i * dimension, dimension);
            float walked = float_l2(row, std::span<const float>(centroids).subspan(std::size_t(labels[sample[i]]) * dimension, dimension));
            float best = float_l2(row, std::span<const float>(centroids).subspan(std::size_t(exact[i]) * dimension, dimension));
            same += labels[sample[i]] == exact[i]; walk_objective += walked; exact_objective += best;
            double relative = best > 0 ? double(walked) / double(best) - 1 : 0;
            if (relative > 1e-6) { ++worse; excess += relative; worst = std::max(worst, relative); }
        }
        if (args.contains("--labels-output")) {
            std::ofstream out(args.at("--labels-output"), std::ios::binary);
            out.write(reinterpret_cast<const char*>(labels.data()), std::streamsize(labels.size() * sizeof(std::uint32_t)));
            if (!out) throw std::runtime_error("cannot write labels");
        }
        std::ostringstream report;
        report << "{\"rows\": " << rows << ", \"k\": " << centroids_count << ", \"supers\": " << options.supers
               << ", \"probes\": " << options.probes << ", \"threads\": " << options.threads << ", \"total_s\": " << total_s
               << ", \"supers_s\": " << stats.supers_s << ", \"assign_s\": " << stats.assign_s
               << ", \"mean_candidates\": " << stats.mean_candidates
               << ", \"check_rows\": " << sample.size() << ", \"same_label\": " << double(same) / double(sample.size())
               << ", \"worse_distance\": " << double(worse) / double(sample.size()) << ", \"mean_excess_when_worse\": " << (worse ? excess / double(worse) : 0)
               << ", \"max_excess\": " << worst << ", \"objective_ratio\": " << walk_objective / exact_objective
               << ", \"brute_force_sample_s\": " << brute_s << "}\n";
        std::cout << report.str();
        if (args.contains("--report")) std::ofstream(args.at("--report")) << report.str();
    } catch (const std::exception& error) { std::cerr << "error: " << error.what() << '\n'; return 1; }
}
