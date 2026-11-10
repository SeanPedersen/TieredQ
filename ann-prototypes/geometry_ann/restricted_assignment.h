// Nearest-centroid assignment restricted to the members of each row's P nearest super-centroids (k-means over the
// centroids). Rows probing the same super-centroid are scored together with one GEMM, so the work is a few percent of
// exhaustive assignment while staying BLAS-shaped. Exactness is measured against brute force.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>
namespace geometry_ann {
struct SuperCentroids {
    std::vector<float> centers;                       // supers x dimension
    std::vector<std::vector<std::uint32_t>> members;  // centroid ids per super-centroid
};
struct AssignmentOptions {
    std::size_t supers = 256, probes = 16, super_iterations = 6, threads = 1, chunk_rows = 65536;
    std::uint32_t seed = 42;
};
struct AssignmentStats {
    double supers_s = 0, assign_s = 0, mean_candidates = 0;
};
// Writes rows [first, first + count) as float into out (count x dimension).
using DecodeRows = std::function<void(std::size_t first, std::size_t count, std::span<float> out)>;
// Squared L2 in dot-product form ||x||^2 + ||c||^2 - 2 x.c for all pairs: out is rows x centroids.
void squared_distances(std::span<const float> rows, std::span<const float> centroids, std::span<const float> centroid_norms,
                       std::size_t dimension, std::span<float> out);
SuperCentroids cluster_centroids(std::span<const float> centroids, std::size_t dimension, const AssignmentOptions& options);
AssignmentStats restricted_assign(std::size_t rows, std::span<const float> centroids, std::size_t dimension,
                                  const AssignmentOptions& options, const DecodeRows& decode, std::span<std::uint32_t> labels);
// Exact nearest centroid (lowest id on ties); the exactness reference and the super-centroid assignment.
std::vector<std::uint32_t> brute_force_assign(std::span<const float> rows, std::span<const float> centroids,
                                              std::size_t dimension, std::size_t threads);
}
