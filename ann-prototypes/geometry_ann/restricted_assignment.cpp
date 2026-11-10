// Grouped restricted assignment: one GEMM picks each row's top-P super-centroids; each super-centroid then scores the
// rows that probed it against its members with one GEMM; every row keeps the best of its P partial minima.
#include "restricted_assignment.h"
#include "parallel.h"
#include <cblas.h>
#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
namespace geometry_ann {
namespace {
constexpr std::size_t brute_force_block = 1024;
std::vector<float> norms(std::span<const float> points, std::size_t dimension) {
    std::vector<float> result(points.size() / dimension);
    for (std::size_t i = 0; i < result.size(); ++i) {
        float sum = 0; for (std::size_t j = 0; j < dimension; ++j) sum += points[i * dimension + j] * points[i * dimension + j];
        result[i] = sum;
    }
    return result;
}
std::uint32_t nearest(const float* distances, std::size_t count) {
    return std::uint32_t(std::min_element(distances, distances + count) - distances);
}
struct Partial { float distance; std::uint32_t centroid; };
}
void squared_distances(std::span<const float> rows, std::span<const float> centroids, std::span<const float> centroid_norms,
                       std::size_t dimension, std::span<float> out) {
    const std::size_t m = rows.size() / dimension, n = centroids.size() / dimension;
    if (out.size() < m * n || centroid_norms.size() != n) throw std::invalid_argument("distance shapes");
    if (!m || !n) return;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(m), int(n), int(dimension), -2.f, rows.data(), int(dimension),
                centroids.data(), int(dimension), 0.f, out.data(), int(n));
    auto row_norms = norms(rows, dimension);
    for (std::size_t i = 0; i < m; ++i) for (std::size_t j = 0; j < n; ++j) out[i * n + j] += row_norms[i] + centroid_norms[j];
}
std::vector<std::uint32_t> brute_force_assign(std::span<const float> rows, std::span<const float> centroids,
                                              std::size_t dimension, std::size_t threads) {
    const std::size_t m = rows.size() / dimension, n = centroids.size() / dimension;
    auto centroid_norms = norms(centroids, dimension);
    std::vector<std::uint32_t> labels(m);
    openblas_set_num_threads(1);
    parallel_for((m + brute_force_block - 1) / brute_force_block, threads, [&](std::size_t block, std::size_t) {
        const std::size_t first = block * brute_force_block, count = std::min(brute_force_block, m - first);
        std::vector<float> distances(count * n);
        squared_distances(rows.subspan(first * dimension, count * dimension), centroids, centroid_norms, dimension, distances);
        for (std::size_t r = 0; r < count; ++r) labels[first + r] = nearest(distances.data() + r * n, n);
    });
    return labels;
}
SuperCentroids cluster_centroids(std::span<const float> centroids, std::size_t dimension, const AssignmentOptions& options) {
    const std::size_t count = centroids.size() / dimension, supers = std::min(options.supers, count);
    std::vector<std::uint32_t> pick(count); std::iota(pick.begin(), pick.end(), 0u);
    std::mt19937 rng(options.seed); std::shuffle(pick.begin(), pick.end(), rng);
    SuperCentroids result; result.centers.resize(supers * dimension);
    for (std::size_t s = 0; s < supers; ++s)
        std::copy_n(centroids.begin() + std::ptrdiff_t(pick[s] * dimension), dimension, result.centers.begin() + std::ptrdiff_t(s * dimension));
    std::vector<std::uint32_t> owner;
    for (std::size_t iteration = 0;; ++iteration) {
        owner = brute_force_assign(centroids, result.centers, dimension, options.threads);
        if (iteration == options.super_iterations) break;
        std::vector<double> sums(supers * dimension, 0); std::vector<std::size_t> sizes(supers, 0);
        for (std::size_t c = 0; c < count; ++c) {
            ++sizes[owner[c]];
            for (std::size_t j = 0; j < dimension; ++j) sums[owner[c] * dimension + j] += centroids[c * dimension + j];
        }
        for (std::size_t s = 0; s < supers; ++s) if (sizes[s])
            for (std::size_t j = 0; j < dimension; ++j) result.centers[s * dimension + j] = float(sums[s * dimension + j] / double(sizes[s]));
    }
    result.members.assign(supers, {});
    for (std::size_t c = 0; c < count; ++c) result.members[owner[c]].push_back(std::uint32_t(c));
    return result;
}
AssignmentStats restricted_assign(std::size_t rows, std::span<const float> centroids, std::size_t dimension,
                                  const AssignmentOptions& options, const DecodeRows& decode, std::span<std::uint32_t> labels) {
    using Clock = std::chrono::steady_clock;
    auto seconds = [](Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); };
    if (labels.size() != rows) throw std::invalid_argument("one label per row required");
    AssignmentStats stats;
    auto start = Clock::now();
    auto supers = cluster_centroids(centroids, dimension, options);
    stats.supers_s = seconds(start);
    const std::size_t super_count = supers.members.size(), probes = std::min(options.probes, super_count);
    auto super_norms = norms(supers.centers, dimension), centroid_norms = norms(centroids, dimension);
    // Each super-centroid's members are packed contiguously so one GEMM scores a group of rows against all of them.
    std::vector<std::vector<float>> member_vectors(super_count), member_norms(super_count);
    for (std::size_t s = 0; s < super_count; ++s) for (auto c : supers.members[s]) {
        member_vectors[s].insert(member_vectors[s].end(), centroids.begin() + std::ptrdiff_t(c * dimension), centroids.begin() + std::ptrdiff_t((c + 1) * dimension));
        member_norms[s].push_back(centroid_norms[c]);
    }
    start = Clock::now();
    openblas_set_num_threads(1);
    std::size_t candidates = 0;
    std::vector<float> block(options.chunk_rows * dimension);
    for (std::size_t first = 0; first < rows; first += options.chunk_rows) {
        const std::size_t chunk = std::min(options.chunk_rows, rows - first);
        auto rows_view = std::span<float>(block).first(chunk * dimension);
        decode(first, chunk, rows_view);
        std::vector<std::uint32_t> probed(chunk * probes);
        parallel_for((chunk + brute_force_block - 1) / brute_force_block, options.threads, [&](std::size_t b, std::size_t) {
            const std::size_t begin = b * brute_force_block, count = std::min(brute_force_block, chunk - begin);
            std::vector<float> distances(count * super_count); std::vector<std::uint32_t> order(super_count);
            squared_distances(std::span<const float>(rows_view).subspan(begin * dimension, count * dimension), supers.centers, super_norms, dimension, distances);
            for (std::size_t r = 0; r < count; ++r) {
                const float* d = distances.data() + r * super_count;
                std::iota(order.begin(), order.end(), 0u);
                std::partial_sort(order.begin(), order.begin() + std::ptrdiff_t(probes), order.end(),
                                  [&](std::uint32_t a, std::uint32_t c) { return d[a] < d[c] || (d[a] == d[c] && a < c); });
                std::copy_n(order.begin(), probes, probed.begin() + std::ptrdiff_t((begin + r) * probes));
            }
        });
        // Group (row, probe slot) pairs by super-centroid; each slot is written by exactly one group.
        std::vector<std::vector<std::uint32_t>> groups(super_count);
        for (std::size_t i = 0; i < chunk * probes; ++i) groups[probed[i]].push_back(std::uint32_t(i));
        std::vector<Partial> partial(chunk * probes);
        parallel_for(super_count, options.threads, [&](std::size_t s, std::size_t) {
            const auto& slots = groups[s]; const std::size_t members = supers.members[s].size();
            if (slots.empty() || !members) {
                for (auto slot : slots) partial[slot] = {std::numeric_limits<float>::infinity(), std::numeric_limits<std::uint32_t>::max()};
                return;
            }
            std::vector<float> gathered(slots.size() * dimension), distances(slots.size() * members);
            for (std::size_t i = 0; i < slots.size(); ++i)
                std::copy_n(rows_view.begin() + std::ptrdiff_t((slots[i] / probes) * dimension), dimension, gathered.begin() + std::ptrdiff_t(i * dimension));
            squared_distances(gathered, member_vectors[s], member_norms[s], dimension, distances);
            for (std::size_t i = 0; i < slots.size(); ++i) {
                const float* d = distances.data() + i * members;
                std::size_t best = 0;
                for (std::size_t m = 1; m < members; ++m)
                    if (d[m] < d[best] || (d[m] == d[best] && supers.members[s][m] < supers.members[s][best])) best = m;
                partial[slots[i]] = {d[best], supers.members[s][best]};
            }
        });
        for (std::size_t r = 0; r < chunk; ++r) {
            Partial best = partial[r * probes];
            for (std::size_t p = 0; p < probes; ++p) {
                const auto& option = partial[r * probes + p];
                candidates += supers.members[probed[r * probes + p]].size();
                if (option.distance < best.distance || (option.distance == best.distance && option.centroid < best.centroid)) best = option;
            }
            labels[first + r] = best.centroid;
        }
    }
    stats.assign_s = seconds(start);
    stats.mean_candidates = double(candidates) / double(rows);
    return stats;
}
}
