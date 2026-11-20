// Two-level TieredQ partition (see partition.hpp).
#include "partition.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>

#include <omp.h>

#include "tieredq_train_emeans.h"

namespace hh {
namespace {

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// CSR grouping of rows by label: members[offsets[c]..offsets[c+1]) are label c's rows, ascending
void group_by(const uint32_t* labels, size_t n, size_t n_groups, std::vector<int64_t>& offsets,
              std::vector<uint32_t>& members) {
    offsets.assign(n_groups + 1, 0);
    for (size_t i = 0; i < n; ++i) ++offsets[labels[i] + 1];
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    members.resize(n);
    std::vector<int64_t> cursor(offsets.begin(), offsets.end() - 1);
    for (size_t i = 0; i < n; ++i) members[cursor[labels[i]]++] = static_cast<uint32_t>(i);
}

std::vector<float> mean_row(const float* rows, size_t n, size_t dim) {
    std::vector<float> mean(dim, 0.0f);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < dim; ++j) mean[j] += rows[i * dim + j];
    for (float& v : mean) v /= static_cast<float>(n);
    return mean;
}

}  // namespace

Partition::Partition(const float* rows, size_t n_rows, size_t dim, const PartitionConfig& config) : dim_(dim) {
    if (config.coarse_lists == 0 || config.fine_per_coarse == 0) throw std::invalid_argument("empty partition");
    if (n_rows < config.coarse_lists) throw std::invalid_argument("fewer rows than coarse lists");
    auto start = std::chrono::steady_clock::now();
    const auto assignment = train_coarse(rows, n_rows, config);
    stats.coarse_s = seconds_since(start);

    start = std::chrono::steady_clock::now();
    train_fine(rows, assignment, config);
    stats.fine_s = seconds_since(start);
}

// TieredQIVFFlatIndex's list training without its index (no FP32 copy of the rows): e-means streamed from a seeded
// sample of sample_rows_per_list rows per list, then exact BLAS assignment of every row to the binary16-rounded
// centroids.
std::vector<uint32_t> Partition::train_coarse(const float* rows, size_t n_rows, const PartitionConfig& config) {
    namespace tqd = tq::tieredq_detail;
    const size_t lists = config.coarse_lists;
    const size_t sample_rows = std::min(n_rows, lists * config.sample_rows_per_list);
    auto picked = tqd::TrainingRng(config.seed).choice(n_rows, sample_rows);
    std::sort(picked.begin(), picked.end());
    const tqd::TrainingRowSource source = [&](const uint32_t* positions, size_t n, float* out) {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) std::copy_n(rows + picked[positions[i]] * dim_, dim_, out + i * dim_);
    };
    tq::TieredQCoarseTraining training;
    training.assignment_supers = 0;
    training.sample_rows_per_list = config.sample_rows_per_list;
    training.emeans_epochs = config.coarse_epochs;
    const auto coarse = tqd::train_coarse_rows(source, sample_rows, dim_, lists, config.seed, training);
    coarse_centroids = coarse.centroids();
    std::vector<uint32_t> assignment(n_rows);
    tqd::BlasAssigner(coarse).assign(rows, n_rows, assignment.data());
    return assignment;
}

// Trains one coarse list's fine cells: writes its rows' stored positions, fine sizes and (padded) fine centroids.
void Partition::train_list(size_t list, const float* rows, const uint32_t* list_rows, const PartitionConfig& config) {
    const size_t m = config.fine_per_coarse;
    const int64_t first = coarse_offsets[list];
    const size_t count = static_cast<size_t>(coarse_offsets[list + 1] - first);
    std::vector<float> cell(count * dim_), centroids, half_norms;
    for (size_t i = 0; i < count; ++i)
        std::copy_n(rows + static_cast<size_t>(list_rows[i]) * dim_, dim_, cell.data() + i * dim_);

    const size_t k = std::min(m, count);
    std::vector<uint32_t> labels(count, 0), fine_members;
    if (k >= 2) {
        centroids = tq::tieredq_detail::train_emeans(cell.data(), count, dim_, k, config.seed + list, config.fine_epochs);
        half_norms.resize(k);
        for (size_t c = 0; c < k; ++c) {
            const float* centroid = centroids.data() + c * dim_;
            half_norms[c] = 0.5f * std::inner_product(centroid, centroid + dim_, centroid, 0.0f);
        }
        tq::tieredq_detail::nearest_centroids(centroids.data(), half_norms.data(), k, dim_, cell.data(), count,
                                              labels.data());
    } else if (k == 1) {
        centroids = mean_row(cell.data(), count, dim_);
    } else {
        centroids.assign(coarse_centroids.begin() + static_cast<int64_t>(list * dim_),
                         coarse_centroids.begin() + static_cast<int64_t>((list + 1) * dim_));
    }

    const size_t trained = std::max<size_t>(k, 1);
    std::vector<int64_t> fine_local;
    group_by(labels.data(), count, trained, fine_local, fine_members);
    for (size_t i = 0; i < count; ++i) {
        const uint32_t row = list_rows[fine_members[i]];
        order[first + static_cast<int64_t>(i)] = row;
    }
    for (size_t slot = 0; slot < m; ++slot) {
        const size_t source = slot % trained;
        const size_t size = slot < trained ? static_cast<size_t>(fine_local[source + 1] - fine_local[source]) : 0;
        fine_offsets[list * m + slot + 1] = static_cast<int64_t>(size);
        std::copy_n(centroids.data() + source * dim_, dim_, fine_centroids.data() + (list * m + slot) * dim_);
    }
}

void Partition::train_fine(const float* rows, const std::vector<uint32_t>& assignment, const PartitionConfig& config) {
    const size_t n_coarse = config.coarse_lists, m = config.fine_per_coarse, n = assignment.size();
    std::vector<uint32_t> members;
    group_by(assignment.data(), n, n_coarse, coarse_offsets, members);
    order.resize(n);
    fine_offsets.assign(n_coarse * m + 1, 0);
    fine_centroids.resize(n_coarse * m * dim_);

    // One list per thread, each trained single-threaded: lists are small (~300 rows), so per-list parallel regions
    // cost more than they compute (1024 lists in sequence: 136 s). OpenBLAS is pinned to one thread first, so the
    // trainer's own BlasThreads guards (sized by each worker's OpenMP budget of 1) only ever set it to 1.
    const tq::tieredq_detail::BlasThreads single_blas(1);
#pragma omp parallel
    {
        omp_set_num_threads(1);
#pragma omp for schedule(dynamic, 1)
        for (size_t list = 0; list < n_coarse; ++list)
            train_list(list, rows, members.data() + coarse_offsets[list], config);
    }
    std::partial_sum(fine_offsets.begin(), fine_offsets.end(), fine_offsets.begin());
}

}  // namespace hh
