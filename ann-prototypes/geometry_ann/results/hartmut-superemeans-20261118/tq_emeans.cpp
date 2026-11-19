// TieredQ's C++ e-means (EmeansTrainer) inside the SuperEMeans reproducibility harness: same data, ground truth,
// final assignment (SuperKMeans::Assign, exact fp32), WCSS and IVF recall evaluator as ./emeans.
// Usage: tq_emeans <data_dir> <dataset> <n_clusters> <supers: 0 = exact> <probes> [epochs=2]
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "bench_utils.h"
#include "peak_rss.h"
#include "superkmeans/superkmeans.h"
#include "tieredq_train_emeans.h"

int main(int argc, char* argv[]) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <data_dir> <dataset> <n_clusters> <supers> <probes> [epochs=2]\n", argv[0]);
        return 1;
    }
    const std::string data_dir = argv[1], dataset = argv[2];
    const size_t n_clusters = std::stoul(argv[3]), supers = std::stoul(argv[4]), probes = std::stoul(argv[5]);
    const double epochs = argc > 6 ? std::stod(argv[6]) : 2.0;
    const auto [n, d] = repro::GetDataset(dataset);
    const std::string algorithm = supers ? "tq_emeans_s" + std::to_string(supers) + "p" + std::to_string(probes)
                                         : "tq_emeans_exact";
    constexpr uint64_t seed = 42;

    printf("=== %s | %s (n=%zu, d=%zu) | k=%zu | %g epochs ===\n", algorithm.c_str(), dataset.c_str(), n, d,
           n_clusters, epochs);
    auto data = repro::ReadFloats(repro::DataPath(data_dir, dataset), n * d);
    auto queries = repro::ReadFloats(repro::QueriesPath(data_dir, dataset), repro::N_QUERIES * d);
    const auto ground_truth = repro::ReadGroundTruth(repro::GroundTruthPath(data_dir, dataset));

    const auto recipe = tq::tieredq_detail::emeans_recipe(n, n_clusters, d, epochs);
    printf("Batch size: %zu | Initial step: %f\n", recipe.batch_rows, recipe.alpha0);
    const repro::PeakRss rss;
    repro::Timer timer;
    timer.Tic();
    const std::vector<float> centroids =
        tq::tieredq_detail::train_emeans(data.get(), n, d, n_clusters, seed, epochs, supers, probes);
    timer.Toc();
    printf("\nTraining completed in %.1f s\n", timer.Seconds());
    rss.Print();

    skmeans::SuperKMeansConfig assign_config;
    assign_config.n_threads = repro::Threads();
    skmeans::SuperKMeans<skmeans::Quantization::f32> assigner(n_clusters, d, assign_config);
    const auto assignments = assigner.Assign(data.get(), centroids.data(), n, n_clusters);

    repro::RecallEvaluator evaluator(ground_truth, queries.get(), repro::N_QUERIES, centroids.data(), n_clusters, d);
    const double wcss = skmeans::SuperKMeans<>::ComputeWCSS(data.get(), centroids.data(), assignments.data(), n, d);
    printf("\n  [assign labels] WCSS: %.2f\n", wcss);
    repro::PrintRecall(evaluator.Evaluate(assignments, 10), 10);
    const auto recall_at_100 = evaluator.Evaluate(assignments, 100);
    repro::PrintRecall(recall_at_100, 100);
    repro::PrintResult(algorithm, dataset, n_clusters, uint32_t(epochs), "assign", timer.Seconds(), wcss,
                       recall_at_100);
    return 0;
}
