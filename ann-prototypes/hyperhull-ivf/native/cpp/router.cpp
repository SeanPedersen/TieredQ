// Combined hull router search (see router.hpp for the scoring model).
#include "router.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kernels.hpp"
#include "select.hpp"

namespace hh {

using Clock = std::chrono::steady_clock;
using Scored = std::pair<float, uint32_t>;

// Per-thread buffers; scores and ids are kept as separate contiguous arrays for the selector.
struct Scratch {
    std::vector<float> basis_strip, zq, residual_q, coarse_score, fine_score, cand_score, refine_score;
    std::vector<int8_t> weights;
    std::vector<int32_t> idots, cand_idot;
    std::vector<uint32_t> cand_row, coarse_sel, fine_ids, fine_sel, cand_sel, refine_pos;
    std::vector<uint64_t> mask;  // bitmap for sort_by_bitmap (all zero between uses)
    std::vector<Scored> best;
    std::vector<uint64_t> seen;  // row bitmap for graph expansion (all zero between uses)
    std::vector<uint32_t> touched, expand_rows, expand_sel;
    std::vector<float> expand_score;
    SelectScratch select;
};

namespace {

constexpr size_t kBlockDims = 64;
constexpr float kInt8Max = 127.0f;
#ifndef HH_PROJ_CHUNK
#define HH_PROJ_CHUNK 256
#endif
constexpr size_t kProjChunkDims = HH_PROJ_CHUNK;  // basis dims per L1-resident strip chunk (16 KB fp32)
#ifndef HH_CELLS_AHEAD
#define HH_CELLS_AHEAD 3
#endif
#ifndef HH_RERANK_AHEAD
#define HH_RERANK_AHEAD 3
#endif
#ifndef HH_RECORDS_AHEAD
#define HH_RECORDS_AHEAD 32
#endif
constexpr size_t kGroupRows = 4;                    // sketch records per kernel call
constexpr size_t kRecordsAhead = HH_RECORDS_AHEAD;  // pass-1 prefetch distance (records, multiple of kGroupRows)
constexpr size_t kRefineAhead = 8;               // pass-2 prefetch distance (suffix records)
constexpr size_t kCellsAhead = HH_CELLS_AHEAD;    // fine-stage prefetch distance (coarse cells)
constexpr size_t kRerankAhead = HH_RERANK_AHEAD;  // exact-rerank prefetch distance (vectors)

Scratch& scratch() {
    thread_local Scratch s;
    return s;
}

uint32_t elapsed_ns(Clock::time_point from, Clock::time_point to, size_t share = 1) {
    return static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count() / share);
}

struct QueryWeights {
    float scale, offset;  // <z_q, ẑ> ≈ offset + scale · Σ w_d·code_d
};

QueryWeights quantize_query(const float* zq, const float* lo, const float* step, size_t dims, int8_t* w) {
    float max_abs = 0.0f, offset = 0.0f;
    for (size_t d = 0; d < dims; ++d) {
        max_abs = std::max(max_abs, std::abs(zq[d] * step[d]));
        offset += zq[d] * lo[d];
    }
    const float scale = max_abs > 0.0f ? max_abs / kInt8Max : 1.0f;
    for (size_t d = 0; d < dims; ++d) w[d] = static_cast<int8_t>(std::lrintf(zq[d] * step[d] / scale));
    return {scale, offset};
}

void check_dims(size_t dims, const char* what) {
    if (dims == 0 || dims % kBlockDims != 0) throw std::invalid_argument(std::string(what) + " must be a multiple of 64");
}

// 4-bit code of dim d in a packed row (see kernels.hpp for the layout)
float code4(const uint8_t* row, size_t d) {
    const uint8_t byte = row[(d / kBlockDims) * (kBlockDims / 2) + d % (kBlockDims / 2)];
    return static_cast<float>((d % kBlockDims) < kBlockDims / 2 ? byte & 0x0F : byte >> 4);
}

// Visit order of one coarse cell's m fine cells: a greedy nearest-neighbour chain over their
// (dequantized) centroids, starting at the one farthest from their mean; empty cells go last.
// Fine cells that are close in space tend to be selected together, so they become one contiguous read.
std::vector<uint32_t> chain_order(const std::vector<float>& centroids, const std::vector<bool>& empty, size_t dims) {
    const size_t m = empty.size();
    std::vector<float> mean(dims, 0.0f);
    size_t n_full = 0;
    for (size_t i = 0; i < m; ++i) {
        if (empty[i]) continue;
        ++n_full;
        for (size_t d = 0; d < dims; ++d) mean[d] += centroids[i * dims + d];
    }
    for (auto& v : mean) v /= static_cast<float>(std::max<size_t>(n_full, 1));
    auto dist = [&](const float* a, const float* b) {
        float sum = 0.0f;
        for (size_t d = 0; d < dims; ++d) sum += (a[d] - b[d]) * (a[d] - b[d]);
        return sum;
    };
    std::vector<uint32_t> order;
    std::vector<bool> used(empty);
    size_t current = m;
    float farthest = -1.0f;
    for (size_t i = 0; i < m; ++i)
        if (!empty[i] && dist(&centroids[i * dims], mean.data()) > farthest)
            farthest = dist(&centroids[i * dims], mean.data()), current = i;
    while (current < m) {
        order.push_back(static_cast<uint32_t>(current));
        used[current] = true;
        size_t next = m;
        float nearest = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < m; ++i)
            if (!used[i] && dist(&centroids[i * dims], &centroids[current * dims]) < nearest)
                nearest = dist(&centroids[i * dims], &centroids[current * dims]), next = i;
        current = next;
    }
    for (size_t i = 0; i < m; ++i)
        if (empty[i]) order.push_back(static_cast<uint32_t>(i));
    return order;
}

[[maybe_unused]] std::vector<uint32_t> identity_order(size_t n) {  // HH_NO_REORDER A/B builds
    std::vector<uint32_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = static_cast<uint32_t>(i);
    return order;
}

// Sorts distinct values < universe ascending via a bitmap (cheaper than a sort for dense selections).
void sort_by_bitmap(std::vector<uint32_t>& values, size_t universe, std::vector<uint64_t>& mask) {
    mask.resize(std::max(mask.size(), (universe + 63) / 64));
    for (const uint32_t v : values) mask[v / 64] |= uint64_t{1} << (v % 64);
    values.clear();
    for (size_t w = 0; w < (universe + 63) / 64; ++w) {
        for (uint64_t bits = mask[w]; bits; bits &= bits - 1) values.push_back(static_cast<uint32_t>(w * 64 + std::countr_zero(bits)));
        mask[w] = 0;
    }
}

// Pass-1 scan over candidate sketch records (count a multiple of kGroupRows): idot and score
// aux[0] + offset + scale·idot - resid_weight·aux[1] of each, 4 records per kernel call.
struct PrefixScan {
    const uint8_t* codes;
    size_t row_bytes;
    const float* aux;
    size_t aux_floats;
    const uint32_t* cand;
    size_t count;
    float offset, scale, resid_weight;
    int32_t* idots;
    float* scores;
};

template <class Dot4>
void scan_prefix(const PrefixScan& p, const Dot4 dot4) {
    auto codes_of = [&](size_t i) { return p.codes + static_cast<size_t>(p.cand[i]) * p.row_bytes; };
    auto aux_of = [&](size_t i) { return p.aux + static_cast<size_t>(p.cand[i]) * p.aux_floats; };
    auto prefetch_group = [&](size_t i) {
        for (size_t r = 0; r < kGroupRows; ++r) prefetch(codes_of(i + r), p.row_bytes);
        _mm_prefetch(reinterpret_cast<const char*>(aux_of(i)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(aux_of(i + kGroupRows - 1) + p.aux_floats - 1), _MM_HINT_T0);
    };
    const __m128 offset = _mm_set1_ps(p.offset), scale = _mm_set1_ps(p.scale), resid_weight = _mm_set1_ps(p.resid_weight);
    for (size_t i = 0; i < std::min(kRecordsAhead, p.count); i += kGroupRows) prefetch_group(i);
    for (size_t i = 0; i < p.count; i += kGroupRows) {
        if (i + kRecordsAhead < p.count) prefetch_group(i + kRecordsAhead);
        const __m128i idot = dot4(codes_of(i), codes_of(i + 1), codes_of(i + 2), codes_of(i + 3));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p.idots + i), idot);
        // [bias, o_P] of the 4 records → bias and o_P lanes
        const __m128 aux01 = _mm_loadh_pi(_mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(aux_of(i))),
                                          reinterpret_cast<const __m64*>(aux_of(i + 1)));
        const __m128 aux23 = _mm_loadh_pi(_mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(aux_of(i + 2))),
                                          reinterpret_cast<const __m64*>(aux_of(i + 3)));
        const __m128 bias = _mm_shuffle_ps(aux01, aux23, _MM_SHUFFLE(2, 0, 2, 0));
        const __m128 prefix_resid = _mm_shuffle_ps(aux01, aux23, _MM_SHUFFLE(3, 1, 3, 1));
        _mm_storeu_ps(p.scores + i, _mm_fnmadd_ps(resid_weight, prefix_resid, _mm_fmadd_ps(scale, _mm_cvtepi32_ps(idot), _mm_add_ps(bias, offset))));
    }
}

// Exact squared distances of the given vector rows, appended to `out` (DRAM-bound: prefetched ahead).
void append_exact(const float* q, const uint32_t* rows, size_t count, const float* vectors, size_t dim, std::vector<Scored>& out) {
    const size_t vector_bytes = dim * sizeof(float);
    auto vector_of = [&](size_t i) { return vectors + static_cast<size_t>(rows[i]) * dim; };
    for (size_t i = 0; i < std::min(kRerankAhead, count); ++i) prefetch(vector_of(i), vector_bytes);
    for (size_t i = 0; i < count; ++i) {
        if (i + kRerankAhead < count) prefetch(vector_of(i + kRerankAhead), vector_bytes);
        out.emplace_back(l2sq(q, vector_of(i), dim), rows[i]);
    }
}

// Runs work(begin, end) over contiguous row ranges on n_threads threads.
template <class Work>
void parallel_rows(size_t n, size_t n_threads, const Work& work) {
    n_threads = std::max<size_t>(1, std::min(n_threads, n));
    std::vector<std::jthread> workers;
    const size_t chunk = (n + n_threads - 1) / n_threads;
    for (size_t t = 0; t < n_threads && t * chunk < n; ++t) workers.emplace_back(work, t * chunk, std::min(n, (t + 1) * chunk));
}

}  // namespace

void Router::build_layout() {
    const size_t m = ix_.fine_per_coarse, n_fine = ix_.n_coarse * m;
    std::vector<float> coarse(ix_.coarse_z, ix_.coarse_z + ix_.n_coarse * ix_.coarse_dims);
    std::vector<bool> coarse_empty(ix_.n_coarse);
    for (size_t c = 0; c < ix_.n_coarse; ++c) coarse_empty[c] = ix_.coarse_offsets[c + 1] == ix_.coarse_offsets[c];
#ifdef HH_NO_REORDER
    coarse_at_ = identity_order(ix_.n_coarse);
#else
    coarse_at_ = chain_order(coarse, coarse_empty, ix_.coarse_dims);
#endif

    fine_at_.resize(n_fine);
    fine_pos_.resize(n_fine);
    std::vector<float> centroids(m * ix_.fine_dims);
    std::vector<bool> empty(m);
    for (size_t slot = 0; slot < ix_.n_coarse; ++slot) {
        const size_t c = coarse_at_[slot];
        for (size_t i = 0; i < m; ++i) {
            const size_t f = c * m + i;
            empty[i] = ix_.fine_offsets[f + 1] == ix_.fine_offsets[f];
            for (size_t d = 0; d < ix_.fine_dims; ++d)
                centroids[i * ix_.fine_dims + d] = ix_.fine_lo[d] + ix_.fine_step[d] * code4(ix_.fine_codes + f * (ix_.fine_dims / 2), d);
        }
#ifdef HH_NO_REORDER
        const std::vector<uint32_t> order = identity_order(m);
#else
        const std::vector<uint32_t> order = chain_order(centroids, empty, ix_.fine_dims);
#endif
        for (size_t k = 0; k < m; ++k) {
            fine_at_[slot * m + k] = static_cast<uint32_t>(c * m + order[k]);
            fine_pos_[slot * m + order[k]] = static_cast<uint32_t>(slot * m + k);
        }
    }
    record_offsets_.resize(n_fine + 1);
    record_offsets_[0] = 0;
    for (size_t pos = 0; pos < n_fine; ++pos) {
        const uint32_t f = fine_at_[pos];
        record_offsets_[pos + 1] = record_offsets_[pos] + (ix_.fine_offsets[f + 1] - ix_.fine_offsets[f]);
    }
}

Router::Router(const IndexView& index) : ix_(index) {
    check_dims(ix_.pca_dims, "pca_dims");
    check_dims(ix_.fine_dims, "fine_dims");
    if (ix_.coarse_dims > ix_.pca_dims || ix_.fine_dims > ix_.pca_dims)
        throw std::invalid_argument("coarse_dims and fine_dims must not exceed pca_dims");
    if (ix_.coarse_dims % 2 != 0) throw std::invalid_argument("coarse_dims must be even");
    if (ix_.dim % 16 != 0) throw std::invalid_argument("dim must be a multiple of 16");
    fine_used_ = ix_.fine_used_dims ? ix_.fine_used_dims : ix_.fine_dims;
    check_dims(fine_used_, "fine_used_dims");
    if (fine_used_ > ix_.fine_dims) throw std::invalid_argument("fine_used_dims must not exceed fine_dims");
    fine_bytes_ = fine_used_ / 2;
    fine_blocks_ = fine_used_ / kBlockDims;
    if (ix_.sketch_bits != 2 && ix_.sketch_bits != 4) throw std::invalid_argument("sketch_bits must be 2 or 4");
    if (ix_.sketch_bits == 2 && (ix_.pca_dims / 4) % 32 != 0) throw std::invalid_argument("2-bit sketch needs pca_dims % 128 == 0");
    sketch_bytes_ = ix_.pca_dims * ix_.sketch_bits / 8;
    prefix_dims_ = ix_.prefix_dims ? ix_.prefix_dims : ix_.pca_dims;
    check_dims(prefix_dims_, "prefix_dims");
    if (prefix_dims_ > ix_.pca_dims) throw std::invalid_argument("prefix_dims must not exceed pca_dims");
    if (ix_.sketch_bits == 2 && prefix_dims_ != ix_.pca_dims) throw std::invalid_argument("2-bit sketch has no prefix split");
    prefix_bytes_ = prefix_dims_ * ix_.sketch_bits / 8;
    suffix_bytes_ = sketch_bytes_ - prefix_bytes_;
    coarse_padded_ = (ix_.n_coarse + kCentroidBlock - 1) / kCentroidBlock * kCentroidBlock;

    // transposed basis in strips of kStripOutputs outputs: strip k holds proj[16k + o][d] at [d·16 + o]
    proj_strips_.resize(ix_.pca_dims * ix_.dim);
    std::vector<float> strip(ix_.dim * kStripOutputs);
    for (size_t k = 0; k < ix_.pca_dims / kStripOutputs; ++k) {
        for (size_t d = 0; d < ix_.dim; ++d)
            for (size_t o = 0; o < kStripOutputs; ++o) strip[d * kStripOutputs + o] = ix_.proj[(k * kStripOutputs + o) * ix_.dim + d];
        to_f16(strip.data(), strip.size(), proj_strips_.data() + k * strip.size());
    }

    build_layout();  // all search-side arrays below are indexed by coarse slot (layout order)

    static_assert(kCentroidBlock == kStripOutputs, "coarse blocks are projection strips");
    coarse_blocked_.assign(coarse_padded_ * ix_.coarse_dims, 0.0f);  // padding lanes are never selected
    coarse_norm_.resize(ix_.n_coarse);
    for (size_t slot = 0; slot < ix_.n_coarse; ++slot) {
        const float* centroid = ix_.coarse_z + coarse_at_[slot] * ix_.coarse_dims;
        coarse_norm_[slot] = dot(centroid, centroid, ix_.coarse_dims);
        for (size_t d = 0; d < ix_.coarse_dims; ++d)
            coarse_blocked_[(slot / kCentroidBlock) * kCentroidBlock * ix_.coarse_dims + d * kCentroidBlock + slot % kCentroidBlock] =
                centroid[d];
    }

    const size_t m = ix_.fine_per_coarse;
    // fine centroids keep their first fine_used_ dims; the bias is then ||ẑ_f||² over those dims
    const size_t src_fine_bytes = ix_.fine_dims / 2;
    fine_codes_.resize(ix_.n_coarse * m * fine_bytes_);
    fine_bias_.resize(ix_.n_coarse * m);
    for (size_t slot = 0; slot < ix_.n_coarse; ++slot) {
        for (size_t i = 0; i < m; ++i) {
            const size_t f = coarse_at_[slot] * m + i;
            const uint8_t* src = ix_.fine_codes + f * src_fine_bytes;
            std::memcpy(fine_codes_.data() + (slot * m + i) * fine_bytes_, src, fine_bytes_);
            float bias = ix_.fine_bias[f];
            if (fine_used_ < ix_.fine_dims) {
                bias = 0.0f;
                for (size_t d = 0; d < fine_used_; ++d) {
                    const float v = ix_.fine_lo[d] + ix_.fine_step[d] * code4(src, d);
                    bias += v * v;
                }
            }
            fine_bias_[slot * m + i] = ix_.fine_offsets[f + 1] > ix_.fine_offsets[f] ? bias : std::numeric_limits<float>::infinity();
        }
    }

    // Sketch split into prefix codes scanned in pass 1 and suffix codes read only for pass-2 survivors,
    // plus per-record aux [bias, o_P, o]. Prefix residual o_P = sqrt(o² + ||ẑ_suffix||²): the dims a
    // pass-1 score does not see.
    prefix_codes_.resize(ix_.n_vectors * prefix_bytes_);
    suffix_codes_.resize(ix_.n_vectors * suffix_bytes_);
    record_aux_.resize(ix_.n_vectors * kAuxFloats);
    record_row_.resize(ix_.n_vectors);
    row_record_.resize(ix_.n_vectors);
    for (size_t pos = 0; pos < fine_at_.size(); ++pos) {
        const int64_t first_row = ix_.fine_offsets[fine_at_[pos]];
        for (int64_t i = 0; i < record_offsets_[pos + 1] - record_offsets_[pos]; ++i) {
            const size_t r = static_cast<size_t>(first_row + i), rec = static_cast<size_t>(record_offsets_[pos] + i);
            record_row_[rec] = static_cast<uint32_t>(r);
            row_record_[r] = static_cast<uint32_t>(rec);
            const uint8_t* codes = ix_.sketch_codes + r * sketch_bytes_;
            float suffix_norm2 = 0.0f;
            for (size_t d = prefix_dims_; d < ix_.pca_dims; ++d) {
                const float v = ix_.sketch_lo[d] + ix_.sketch_step[d] * code4(codes, d);
                suffix_norm2 += v * v;
            }
            const float resid = ix_.sketch_resid[r];
            float* aux = record_aux_.data() + rec * kAuxFloats;
            aux[0] = ix_.sketch_bias[r], aux[1] = std::sqrt(resid * resid + suffix_norm2), aux[2] = resid;
            std::memcpy(prefix_codes_.data() + rec * prefix_bytes_, codes, prefix_bytes_);
            if (suffix_bytes_ > 0) std::memcpy(suffix_codes_.data() + rec * suffix_bytes_, codes + prefix_bytes_, suffix_bytes_);
        }
    }
}

void Router::route_block(const float* queries, size_t count, Scratch& s, QueryStats* stats) const {
    const IndexView& x = ix_;
    const auto t0 = Clock::now();

    // 1. projection as a GEMM, K-blocked: each kProjChunkDims × 16 piece of a basis strip is converted
    //    to fp32 once per block (L1-resident) and applied to all the block's queries; outputs start at
    //    -proj_offset and accumulate across chunks
    const size_t strip_floats = x.dim * kStripOutputs;
    s.basis_strip.resize(kProjChunkDims * kStripOutputs);
    s.zq.resize(count * x.pca_dims);
    for (size_t j = 0; j < count; ++j)
        for (size_t p = 0; p < x.pca_dims; ++p) s.zq[j * x.pca_dims + p] = -x.proj_offset[p];
    for (size_t k = 0; k < x.pca_dims / kStripOutputs; ++k) {
        float* out = s.zq.data() + k * kStripOutputs;
        for (size_t d0 = 0; d0 < x.dim; d0 += kProjChunkDims) {
            const size_t dims = std::min(kProjChunkDims, x.dim - d0);
            f16_to_f32(proj_strips_.data() + k * strip_floats + d0 * kStripOutputs, dims * kStripOutputs, s.basis_strip.data());
            project_strip_rows(s.basis_strip.data(), dims, queries + d0, x.dim, count, out, x.pca_dims);
        }
    }
    s.residual_q.resize(count);
    for (size_t j = 0; j < count; ++j) {
        const float* zq = s.zq.data() + j * x.pca_dims;
        const float z_norm2 = dot(zq, zq, x.pca_dims);
        s.residual_q[j] = std::sqrt(std::max(l2sq(queries + j * x.dim, x.mean, x.dim) - z_norm2, 0.0f));
    }
    const auto t1 = Clock::now();

    // 2. coarse hull scores ||c||² - 2<z_q, c> (||z_q||² dropped) with the projection GEMM kernel:
    //    each 16-centroid block is a strip, applied to the block's queries kProjQueries at a time
    s.coarse_score.assign(count * coarse_padded_, 0.0f);
    for (size_t first = 0; first < coarse_padded_; first += kCentroidBlock) {
        const float* block = coarse_blocked_.data() + first * x.coarse_dims;
        project_strip_rows(block, x.coarse_dims, s.zq.data(), x.pca_dims, count, s.coarse_score.data() + first, coarse_padded_);
    }
    for (size_t j = 0; j < count; ++j) {
        float* scores = s.coarse_score.data() + j * coarse_padded_;
        for (size_t c = 0; c < x.n_coarse; ++c) scores[c] = coarse_norm_[c] - 2.0f * scores[c];
    }
    const auto t2 = Clock::now();
    for (size_t j = 0; j < count; ++j) {
        stats[j].stage_ns[kProject] = elapsed_ns(t0, t1, count);
        stats[j].stage_ns[kCoarse] = elapsed_ns(t1, t2, count);
    }
}

void Router::finish_query(const float* q, size_t slot, const SearchParams& p, Scratch& s, float* out_dist,
                          int64_t* out_ids, QueryStats& stats) const {
    const IndexView& x = ix_;
    const float* zq = s.zq.data() + slot * x.pca_dims;
    auto mark = Clock::now();
    auto lap = [&](Stage stage) {
        const auto now = Clock::now();
        stats.stage_ns[stage] += elapsed_ns(mark, now);
        mark = now;
    };
    stats.stage_ns[kFine] = stats.stage_ns[kSketch] = stats.stage_ns[kRefine] = stats.stage_ns[kRerank] = stats.stage_ns[kExpand] = 0;

    // coarse shortlist (its selection is counted in the coarse stage)
    select_smallest(s.coarse_score.data() + slot * coarse_padded_, nullptr, x.n_coarse, p.shortlist, s.coarse_sel, s.select);
    sort_by_bitmap(s.coarse_sel, x.n_coarse, s.mask);  // fine codes in address order (neighbouring slots adjoin)
    const size_t n_short = s.coarse_sel.size();
    lap(kCoarse);

    // 3. fine hull: F best fine cells of the shortlist
    s.weights.resize(x.pca_dims);
    const QueryWeights fw = quantize_query(zq, x.fine_lo, x.fine_step, fine_used_, s.weights.data());
    const size_t m = x.fine_per_coarse;
    s.fine_score.resize(n_short * m);
    s.fine_ids.resize(n_short * m);
    s.idots.resize(std::max(m, s.idots.size()));
    const size_t coarse_cell_bytes = m * fine_bytes_;
    for (size_t i = 0; i < std::min(kCellsAhead, n_short); ++i)
        prefetch(fine_codes_.data() + s.coarse_sel[i] * coarse_cell_bytes, coarse_cell_bytes);
    for (size_t i = 0; i < n_short; ++i) {
        if (i + kCellsAhead < n_short) {
            const size_t ahead = s.coarse_sel[i + kCellsAhead];
            prefetch(fine_codes_.data() + ahead * coarse_cell_bytes, coarse_cell_bytes);
            prefetch(fine_bias_.data() + ahead * m, m * sizeof(float));
        }
        const size_t first = s.coarse_sel[i] * m;
        dot_u4_s8_rows(fine_codes_.data() + first * fine_bytes_, m, fine_bytes_, s.weights.data(), fine_blocks_, s.idots.data());
        const float* bias = fine_bias_.data() + first;
        float* out = s.fine_score.data() + i * m;
        uint32_t* out_ids = s.fine_ids.data() + i * m;
        for (size_t j = 0; j < m; ++j) {
            out[j] = bias[j] - 2.0f * (fw.offset + fw.scale * s.idots[j]);
            out_ids[j] = fine_pos_[first + j];
        }
    }
    // selected layout positions, in address order
    select_about(s.fine_score.data(), s.fine_ids.data(), s.fine_score.size(), p.fine_cells, s.fine_sel, s.select);
    sort_by_bitmap(s.fine_sel, fine_bias_.size(), s.mask);
    lap(kFine);

    // 4. member sketch, pass 1: stream the prefix sketch of every member of the selected fine cells.
    //    Selected cells at most p.gap records apart are merged into one span (the gap's members become
    //    extra candidates). Candidates are scored 4 records at a time across cell boundaries; the list
    //    is padded to a multiple of 4 by repeating its last record.
    s.cand_row.clear();
    int64_t span_begin = 0, span_end = -1;
    auto flush_span = [&] {
        if (span_end <= span_begin) return;
        const size_t at = s.cand_row.size();
        s.cand_row.resize(at + static_cast<size_t>(span_end - span_begin));
        std::iota(s.cand_row.begin() + static_cast<std::ptrdiff_t>(at), s.cand_row.end(), static_cast<uint32_t>(span_begin));
    };
    const auto gap = static_cast<int64_t>(p.gap);
    for (const uint32_t pos : s.fine_sel) {
        const int64_t rb = record_offsets_[pos], re = record_offsets_[pos + 1];
        if (rb == re) continue;
        if (span_end >= 0 && rb - span_end <= gap) {
            span_end = re;
            continue;
        }
        flush_span();
        span_begin = rb, span_end = re;
    }
    flush_span();
    const size_t n_cand = s.cand_row.size(), n_padded = (n_cand + kGroupRows - 1) / kGroupRows * kGroupRows;
    if (n_cand > 0) s.cand_row.resize(n_padded, s.cand_row.back());
    s.cand_score.resize(n_padded);
    s.cand_idot.resize(n_padded);

    // one int8 quantization of the query over all sketch dims; pass 1 uses its prefix
    const QueryWeights sw = quantize_query(zq, x.sketch_lo, x.sketch_step, x.pca_dims, s.weights.data());
    float prefix_offset = 0.0f, suffix_norm2 = 0.0f;
    for (size_t d = 0; d < prefix_dims_; ++d) prefix_offset += zq[d] * x.sketch_lo[d];
    for (size_t d = prefix_dims_; d < x.pca_dims; ++d) suffix_norm2 += zq[d] * zq[d];
    const float resid_q = s.residual_q[slot];
    const float score_scale = -2.0f * sw.scale;
    const size_t prefix_blocks = prefix_dims_ / kBlockDims;
    const int8_t* weights = s.weights.data();
    const uint32_t* cand = s.cand_row.data();
    const PrefixScan scan{prefix_codes_.data(), prefix_bytes_, record_aux_.data(), kAuxFloats, cand, n_padded,
                          -2.0f * prefix_offset, score_scale, 2.0f * x.gamma * std::sqrt(resid_q * resid_q + suffix_norm2),
                          s.cand_idot.data(), s.cand_score.data()};
    using Row = const uint8_t*;
    if (x.sketch_bits == 2)
        scan_prefix(scan, [&](Row a, Row b, Row c, Row d) { return dot_u2_s8_4rows(a, b, c, d, weights, x.pca_dims); });
    else if (prefix_blocks == 1)
        scan_prefix(scan, U4Weights<1>(weights));
    else if (prefix_blocks == 2)
        scan_prefix(scan, U4Weights<2>(weights));
    else if (prefix_blocks == 4)
        scan_prefix(scan, U4Weights<4>(weights));
    else
        scan_prefix(scan, [&](Row a, Row b, Row c, Row d) { return dot_u4_s8_4rows(a, b, c, d, weights, prefix_blocks); });
    const bool two_pass = suffix_bytes_ > 0 && p.refine > 0;
    select_smallest(s.cand_score.data(), nullptr, n_cand, two_pass ? std::max(p.refine, p.rerank) : p.rerank, s.cand_sel, s.select);
    lap(kSketch);

    // pass 2: complete the sketch score of the pass-1 survivors with their suffix dims
    if (two_pass) {
        const size_t n_keep = s.cand_sel.size(), suffix_blocks = suffix_bytes_ * 2 / kBlockDims;
        const float full_score_offset = -2.0f * sw.offset, resid_weight = 2.0f * x.gamma * resid_q;
        const int8_t* suffix_weights = weights + prefix_dims_;
        auto record_of = [&](size_t i) { return static_cast<size_t>(cand[s.cand_sel[i]]); };
        auto suffix_of = [&](size_t i) { return suffix_codes_.data() + record_of(i) * suffix_bytes_; };
        for (size_t i = 0; i < std::min(kRefineAhead, n_keep); ++i) prefetch(suffix_of(i), suffix_bytes_);
        s.refine_score.resize(n_keep);
        for (size_t i = 0; i < n_keep; ++i) {
            if (i + kRefineAhead < n_keep) prefetch(suffix_of(i + kRefineAhead), suffix_bytes_);
            const float* aux = record_aux_.data() + record_of(i) * kAuxFloats;
            const int32_t idot = s.cand_idot[s.cand_sel[i]] + dot_u4_s8(suffix_of(i), suffix_weights, suffix_blocks);
            s.refine_score[i] = aux[0] + full_score_offset + score_scale * static_cast<float>(idot) - resid_weight * aux[2];
        }
        s.refine_pos.assign(s.cand_sel.begin(), s.cand_sel.end());
        select_smallest(s.refine_score.data(), s.refine_pos.data(), n_keep, p.rerank, s.cand_sel, s.select);
    }
    for (auto& c : s.cand_sel) c = record_row_[s.cand_row[c]];  // candidate → sketch record → vector row
    lap(kRefine);

    // 5. exact rerank
    s.best.clear();
    append_exact(q, s.cand_sel.data(), s.cand_sel.size(), x.vectors, x.dim, s.best);
    const size_t n_rerank = s.best.size();
    lap(kRerank);

    // 6. neighbour-graph expansion
    stats.expanded = 0;
    if (p.hops > 0 && graph_k_ > 0 && !s.best.empty()) {
        expand(q, p, s, -2.0f * sw.offset, score_scale, 2.0f * x.gamma * resid_q, stats);
        lap(kExpand);
    }
    const size_t n_out = std::min(p.k, s.best.size());
    std::partial_sort(s.best.begin(), s.best.begin() + n_out, s.best.end());
    for (size_t i = 0; i < p.k; ++i) {
        out_dist[i] = i < n_out ? s.best[i].first : std::numeric_limits<float>::infinity();
        out_ids[i] = i >= n_out ? -1 : p.rows ? static_cast<int64_t>(s.best[i].second) : x.ids[s.best[i].second];
    }
    lap(kRerank);
    stats.sketched = static_cast<uint32_t>(n_cand);
    stats.reranked = static_cast<uint32_t>(n_rerank);
}

// Each hop: graph neighbours of the expand_from best exact results that were not seen yet get the full
// sketch score (prefix + suffix codes, all prefetched up front: one round of independent random reads);
// the best expand_exact of them are scored exactly and join the results.
void Router::expand(const float* q, const SearchParams& p, Scratch& s, float full_offset, float score_scale,
                    float resid_weight, QueryStats& stats) const {
    const IndexView& x = ix_;
    s.seen.resize(std::max(s.seen.size(), (x.n_vectors + 63) / 64));
    s.touched.clear();
    auto mark = [&](uint32_t row) {
        uint64_t& word = s.seen[row / 64];
        const uint64_t bit = uint64_t{1} << (row % 64);
        if (word & bit) return false;
        word |= bit;
        s.touched.push_back(row);
        return true;
    };
    for (const auto& b : s.best) mark(b.second);
    const size_t k_use = std::min(p.expand_k, graph_k_), prefix_blocks = prefix_dims_ / kBlockDims;
    const size_t suffix_blocks = suffix_bytes_ * 2 / kBlockDims;
    const int8_t* weights = s.weights.data();
    size_t expanded = 0;
    for (size_t hop = 0; hop < p.hops; ++hop) {
        const size_t from = std::min(p.expand_from, s.best.size());
        std::partial_sort(s.best.begin(), s.best.begin() + from, s.best.end());
        for (size_t i = 0; i < from; ++i) prefetch(graph_.data() + static_cast<size_t>(s.best[i].second) * graph_k_, k_use * sizeof(uint32_t));
        s.expand_rows.clear();
        for (size_t i = 0; i < from; ++i) {
            const uint32_t* neighbours = graph_.data() + static_cast<size_t>(s.best[i].second) * graph_k_;
            for (size_t j = 0; j < k_use; ++j)
                if (mark(neighbours[j])) s.expand_rows.push_back(neighbours[j]);
        }
        const size_t n_new = s.expand_rows.size();
        if (n_new == 0) break;
        expanded += n_new;
        for (const uint32_t row : s.expand_rows) {
            const size_t rec = row_record_[row];
            prefetch(prefix_codes_.data() + rec * prefix_bytes_, prefix_bytes_);
            if (suffix_bytes_ > 0) prefetch(suffix_codes_.data() + rec * suffix_bytes_, suffix_bytes_);
            _mm_prefetch(reinterpret_cast<const char*>(record_aux_.data() + rec * kAuxFloats), _MM_HINT_T0);
        }
        s.expand_score.resize(n_new);
        for (size_t i = 0; i < n_new; ++i) {
            const size_t rec = row_record_[s.expand_rows[i]];
            const uint8_t* prefix = prefix_codes_.data() + rec * prefix_bytes_;
            int32_t idot = 0;
            if (x.sketch_bits == 2) {
                dot_u2_s8_rows(prefix, 1, prefix_bytes_, weights, x.pca_dims, &idot);
            } else {
                idot = dot_u4_s8(prefix, weights, prefix_blocks);
                if (suffix_bytes_ > 0) idot += dot_u4_s8(suffix_codes_.data() + rec * suffix_bytes_, weights + prefix_dims_, suffix_blocks);
            }
            const float* aux = record_aux_.data() + rec * kAuxFloats;
            s.expand_score[i] = aux[0] + full_offset + score_scale * static_cast<float>(idot) - resid_weight * aux[2];
        }
        select_smallest(s.expand_score.data(), nullptr, n_new, p.expand_exact, s.expand_sel, s.select);
        for (auto& sel : s.expand_sel) sel = s.expand_rows[sel];
        append_exact(q, s.expand_sel.data(), s.expand_sel.size(), x.vectors, x.dim, s.best);
    }
    for (const uint32_t row : s.touched) s.seen[row / 64] = 0;
    stats.expanded = static_cast<uint32_t>(expanded);
}

void Router::build_graph(size_t k, const SearchParams& params, size_t n_threads, bool reverse_edges) {
    // with params.hops > 0 the current graph expands the self-search (a rebuild pass that finds the
    // neighbours a plain router search misses)
    SearchParams p = params;
    p.k = k + 1, p.rows = true;
    if (graph_k_ == 0) p.hops = 0;
    const size_t n = ix_.n_vectors;
    std::vector<float> dist(n * p.k);
    std::vector<int64_t> rows(n * p.k);
    std::vector<QueryStats> stats(n);
    search_batch(ix_.vectors, n, p, n_threads, dist.data(), rows.data(), stats.data());
    auto valid = [&](size_t r, size_t j) { return rows[r * p.k + j] >= 0 && static_cast<size_t>(rows[r * p.k + j]) != r; };

    // reverse edges (r found c ⇒ candidate r for c, same distance) in CSR form
    std::vector<size_t> rev_begin(n + 1, 0);
    std::vector<Scored> rev;
    if (reverse_edges) {
        for (size_t r = 0; r < n; ++r)
            for (size_t j = 0; j < p.k; ++j)
                if (valid(r, j)) ++rev_begin[static_cast<size_t>(rows[r * p.k + j]) + 1];
        for (size_t r = 0; r < n; ++r) rev_begin[r + 1] += rev_begin[r];
        rev.resize(rev_begin[n]);
        std::vector<size_t> at(rev_begin.begin(), rev_begin.end() - 1);
        for (size_t r = 0; r < n; ++r)
            for (size_t j = 0; j < p.k; ++j)
                if (valid(r, j)) rev[at[static_cast<size_t>(rows[r * p.k + j])]++] = {dist[r * p.k + j], static_cast<uint32_t>(r)};
    }
    graph_.assign(n * k, 0);
    auto merge = [&](size_t begin, size_t end) {
        std::vector<Scored> cand;
        for (size_t r = begin; r < end; ++r) {
            cand.clear();
            for (size_t j = 0; j < p.k; ++j)
                if (valid(r, j)) cand.emplace_back(dist[r * p.k + j], static_cast<uint32_t>(rows[r * p.k + j]));
            if (reverse_edges)
                cand.insert(cand.end(), rev.begin() + static_cast<std::ptrdiff_t>(rev_begin[r]),
                            rev.begin() + static_cast<std::ptrdiff_t>(rev_begin[r + 1]));
            std::sort(cand.begin(), cand.end());
            uint32_t* out = graph_.data() + r * k;
            size_t filled = 0;
            for (size_t i = 0; i < cand.size() && filled < k; ++i)
                if (std::find(out, out + filled, cand[i].second) == out + filled) out[filled++] = cand[i].second;
            for (; filled < k; ++filled) out[filled] = static_cast<uint32_t>(r);  // self: always already seen
        }
    };
    parallel_rows(n, n_threads, merge);
    graph_k_ = k;
}

void Router::refine_graph(size_t fanout, size_t n_threads) {
    const size_t n = ix_.n_vectors, k = graph_k_;
    if (k == 0) return;
    fanout = std::min(fanout, k);
    std::vector<uint32_t> refined(n * k);
    auto work = [&](size_t begin, size_t end) {
        std::vector<uint32_t> cand;
        std::vector<Scored> scored;
        for (size_t r = begin; r < end; ++r) {
            const uint32_t* own = graph_.data() + r * k;
            cand.assign(own, own + k);
            for (size_t j = 0; j < fanout; ++j) {
                const uint32_t* next = graph_.data() + static_cast<size_t>(own[j]) * k;
                cand.insert(cand.end(), next, next + fanout);
            }
            std::sort(cand.begin(), cand.end());
            cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
            scored.clear();
            const float* v = ix_.vectors + r * ix_.dim;
            for (const uint32_t c : cand)
                if (c != r) scored.emplace_back(l2sq(v, ix_.vectors + static_cast<size_t>(c) * ix_.dim, ix_.dim), c);
            const size_t keep = std::min(k, scored.size());
            std::partial_sort(scored.begin(), scored.begin() + keep, scored.end());
            for (size_t j = 0; j < k; ++j) refined[r * k + j] = j < keep ? scored[j].second : static_cast<uint32_t>(r);
        }
    };
    parallel_rows(n, n_threads, work);
    graph_ = std::move(refined);
}

void Router::set_graph(const uint32_t* neighbours, size_t k) {
    graph_.assign(neighbours, neighbours + ix_.n_vectors * k);
    graph_k_ = k;
}

void Router::search_range(const float* queries, size_t n_queries, const SearchParams& p, float* out_dist,
                          int64_t* out_ids, QueryStats* stats) const {
    Scratch& s = scratch();
    for (size_t first = 0; first < n_queries; first += kQueryBlock) {
        const size_t count = std::min(kQueryBlock, n_queries - first);
        const float* block = queries + first * ix_.dim;
        route_block(block, count, s, stats + first);
        for (size_t j = 0; j < count; ++j) {
            const size_t i = first + j;
            finish_query(block + j * ix_.dim, j, p, s, out_dist + i * p.k, out_ids + i * p.k, stats[i]);
        }
    }
}

void Router::search_batch(const float* queries, size_t n_queries, const SearchParams& p, size_t n_threads,
                          float* out_dist, int64_t* out_ids, QueryStats* stats) const {
    n_threads = std::max<size_t>(1, std::min(n_threads, n_queries));
    if (n_threads == 1) return search_range(queries, n_queries, p, out_dist, out_ids, stats);
    std::vector<std::jthread> workers;
    const size_t chunk = (n_queries + n_threads - 1) / n_threads;
    for (size_t t = 0; t < n_threads; ++t) {
        const size_t begin = t * chunk, end = std::min(n_queries, begin + chunk);
        if (begin >= end) break;
        workers.emplace_back([=, this] {
            search_range(queries + begin * ix_.dim, end - begin, p, out_dist + begin * p.k, out_ids + begin * p.k,
                         stats + begin);
        });
    }
}

}  // namespace hh
