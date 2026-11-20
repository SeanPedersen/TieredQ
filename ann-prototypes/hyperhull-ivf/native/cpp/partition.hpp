// Two-level partition trained with TieredQ: coarse lists as TieredQIVFFlatIndex trains them (e-means on a per-list
// sample, exact BLAS assignment to binary16-rounded centroids), then fine cells by TieredQ's e-means inside each
// coarse list, one list per thread. Rows are laid out coarse-major, fine-minor (the order HullRouter expects).
// TieredQ's IVF-Flat k-NN graph was tried for the expansion graph and lost to the router's own self-search + reverse
// edges (DBpedia-1536: 0.925 graph recall in 15 s vs 0.934 in 2 s, lower recall after expansion).
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace hh {

struct PartitionConfig {
    size_t coarse_lists = 1024;
    size_t fine_per_coarse = 32;
    double coarse_epochs = 2.5;
    uint32_t sample_rows_per_list = 256;
    double fine_epochs = 2.5;
    uint64_t seed = 42;
};

struct PartitionStats {
    double coarse_s = 0, fine_s = 0;
};

class Partition {
   public:
    Partition(const float* rows, size_t n_rows, size_t dim, const PartitionConfig& config);

    // stored position → original row, and the cell boundaries in stored positions
    std::vector<int64_t> order;
    std::vector<int64_t> coarse_offsets;  // coarse_lists + 1
    std::vector<int64_t> fine_offsets;    // coarse_lists · fine_per_coarse + 1
    std::vector<float> coarse_centroids;  // coarse_lists × dim
    // coarse_lists · fine_per_coarse × dim; a list with fewer than fine_per_coarse cells repeats its own centroids
    // cyclically in the empty slots (zero rows), an empty list repeats its coarse centroid
    std::vector<float> fine_centroids;
    PartitionStats stats;
    size_t dim() const { return dim_; }

   private:
    std::vector<uint32_t> train_coarse(const float* rows, size_t n_rows, const PartitionConfig& config);
    void train_fine(const float* rows, const std::vector<uint32_t>& assignment, const PartitionConfig& config);
    void train_list(size_t list, const float* rows, const uint32_t* list_rows, const PartitionConfig& config);

    size_t dim_;
};

}  // namespace hh
