// Combined hull router search: coarse centroids → 4-bit fine hull → 4-bit member sketch → exact rerank.
//
// All arrays are owned by the caller (Python) and viewed read-only. Vectors are stored coarse-major,
// fine-minor, so every fine cell is a contiguous row range [fine_offsets[f], fine_offsets[f+1]).
// The query is projected once into the PCA space (pca_dims); coarse centroids use its first
// coarse_dims coordinates, fine centroids its first fine_dims, member sketches all pca_dims.
// Scores drop query-only constants (they do not change any ranking):
//   fine   f : fine_bias[f]   - 2·<z_q, ẑ_f>                         (fine_bias   = ||ẑ_f||²)
//   sketch x : sketch_bias[x] - 2·<z_q, ẑ_x> - 2γ·o_q·sketch_resid[x]  (sketch_bias = ||ẑ_x||² + o_x²)
// where <z_q, ẑ> = Σ z_q·lo + Σ (z_q·step)·code is computed with int8 query weights.
// Fine cells may be ranked on only their first fine_used_dims. The member sketch can run in two passes:
// pass 1 streams the first prefix_dims of every member of the selected fine cells (the unseen sketch
// dims are folded into the residual, o_P² = o² + ||ẑ_suffix||²), pass 2 completes the score of the best
// `refine` candidates from a separate suffix store, and the best `rerank` of those are scored exactly.
//
// Batches are routed in blocks of kQueryBlock queries: projection and coarse scores are small GEMMs
// (each basis strip / centroid block is loaded once per block); the remaining stages run per query.
#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

namespace hh {

// Cache-line aligned storage for code rows (a 64-byte row is then exactly one line).
template <class T, size_t Align = 64>
struct AlignedAllocator {
    using value_type = T;
    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };
    AlignedAllocator() = default;
    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Align>&) {}
    T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{Align})); }
    void deallocate(T* p, size_t) { ::operator delete(p, std::align_val_t{Align}); }
    bool operator==(const AlignedAllocator&) const { return true; }
};
using AlignedBytes = std::vector<uint8_t, AlignedAllocator<uint8_t>>;

struct IndexView {
    size_t n_vectors, dim, pca_dims, coarse_dims, fine_dims, sketch_bits, n_coarse, fine_per_coarse;
    const float* mean;         // (dim)
    const float* proj;         // (pca_dims, dim)
    const float* proj_offset;  // (pca_dims) = proj · mean
    const float* coarse_z;     // (n_coarse, coarse_dims)
    const uint8_t* fine_codes;  // (n_coarse·m, fine_dims/2)
    const float *fine_lo, *fine_step, *fine_bias;
    const uint8_t* sketch_codes;  // (n_vectors, pca_dims·sketch_bits/8), sketch_bits ∈ {2, 4}
    const float *sketch_lo, *sketch_step, *sketch_bias, *sketch_resid;
    const float* vectors;  // (n_vectors, dim)
    const int64_t* ids;    // (n_vectors) original ids
    const int64_t* coarse_offsets;  // (n_coarse + 1)
    const int64_t* fine_offsets;    // (n_coarse·m + 1)
    float gamma;
    size_t prefix_dims = 0;     // pass-1 sketch dims (0 = all pca_dims, single pass); multiple of 64
    size_t fine_used_dims = 0;  // fine centroid dims used for ranking (0 = all fine_dims); multiple of 64
};

struct SearchParams {
    size_t k, shortlist, fine_cells, rerank;
    size_t refine = 0;  // pass-1 survivors that get the full sketch score (needs prefix_dims < pca_dims)
    size_t gap = 0;     // merge selected fine cells at most this many records apart into one read
    // neighbour-graph expansion (needs a graph): `hops` rounds; each takes the `expand_from` best exactly
    // scored vectors, sketch-scores their first `expand_k` graph neighbours not seen yet and scores the
    // best `expand_exact` of those exactly
    size_t hops = 0, expand_from = 0, expand_k = 0, expand_exact = 0;
    bool rows = false;  // report vector rows instead of original ids (graph build)
};

enum Stage { kProject, kCoarse, kFine, kSketch, kRefine, kRerank, kExpand, kStages };

struct QueryStats {
    uint32_t sketched, reranked, expanded;  // expanded: graph neighbours sketch-scored
    uint32_t stage_ns[kStages];  // project/coarse are the query's share of its routing block
};

struct Scratch;

class Router {
   public:
#ifndef HH_QUERY_BLOCK
#define HH_QUERY_BLOCK 32
#endif
    static constexpr size_t kQueryBlock = HH_QUERY_BLOCK;

    explicit Router(const IndexView& index);
    void search_batch(const float* queries, size_t n_queries, const SearchParams& params, size_t n_threads,
                      float* out_dist, int64_t* out_ids, QueryStats* stats) const;
    // k-NN graph over the vectors (row space, self excluded), found by searching every vector with `params`
    // (expanded by the current graph when params.hops > 0)
    // reverse_edges: also offer r to every c that r found (with the same distance) before keeping the best k
    void build_graph(size_t k, const SearchParams& params, size_t n_threads, bool reverse_edges = false);
    // one NN-descent round: candidates = current neighbours + the first `fanout` neighbours of the first
    // `fanout` neighbours; every candidate gets its exact distance and the best graph_k remain
    void refine_graph(size_t fanout, size_t n_threads);
    // adopt an external graph: n_vectors × k neighbour rows
    void set_graph(const uint32_t* neighbours, size_t k);
    size_t graph_k() const { return graph_k_; }
    const std::vector<uint32_t>& graph() const { return graph_; }

   private:
    void search_range(const float* queries, size_t n_queries, const SearchParams& params, float* out_dist,
                      int64_t* out_ids, QueryStats* stats) const;
    void route_block(const float* queries, size_t count, Scratch& s, QueryStats* stats) const;
    void build_layout();
    void finish_query(const float* q, size_t slot, const SearchParams& p, Scratch& s, float* out_dist,
                      int64_t* out_ids, QueryStats& stats) const;
    void expand(const float* q, const SearchParams& p, Scratch& s, float full_offset, float score_scale,
                float resid_weight, QueryStats& stats) const;

    IndexView ix_;
    size_t fine_used_, fine_bytes_, fine_blocks_, sketch_bytes_, coarse_padded_;
    size_t prefix_dims_, prefix_bytes_, suffix_bytes_;
    // Search-friendly copies built once from the views (small vs. the vectors):
    std::vector<uint16_t> proj_strips_;  // projection basis as fp16, transposed in 16-output strips
    std::vector<float> coarse_blocked_;  // coarse centroids by slot, blocks of kCentroidBlock, dim-major inside a block
    std::vector<float> coarse_norm_;     // ||c||² by slot
    // sketch records in layout order: pass-1 prefix codes, pass-2 suffix codes (aligned rows) and
    // kAuxFloats per record [bias, prefix residual o_P, residual o]
    static constexpr size_t kAuxFloats = 3;
    AlignedBytes prefix_codes_, suffix_codes_;
    std::vector<float> record_aux_;
    // Layout: coarse cells are renumbered into slots and, inside each, fine cells into positions along
    // nearest-neighbour chains of their centroids, so cells selected together are read contiguously.
    // Search-side coarse / fine arrays are indexed by slot (fine: slot·m + original index within the cell).
    std::vector<uint32_t> coarse_at_;           // slot → coarse id
    std::vector<uint8_t> fine_codes_;           // fine_codes by slot
    std::vector<float> fine_bias_;              // fine_bias by slot, +inf for empty fine cells
    std::vector<uint32_t> fine_at_, fine_pos_;  // position → fine id, slot·m + i → position
    std::vector<int64_t> record_offsets_;       // position → first sketch record (n_fine + 1)
    std::vector<uint32_t> record_row_;          // sketch record → vector row
    std::vector<uint32_t> row_record_;          // vector row → sketch record
    std::vector<uint32_t> graph_;               // row → graph_k_ neighbour rows
    size_t graph_k_ = 0;
};

}  // namespace hh
