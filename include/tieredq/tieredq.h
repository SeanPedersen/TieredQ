// Runtime-sized TieredQ. Link dv_tieredq (LAPACK/BLAS); no Python dependency.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "tieredq/detail/tieredq_scan_plan.h"

namespace dv {
namespace tieredq_detail { class StageProfile; struct FastScanCache; struct FastQueryBasis; struct EncoderData; class LazyEncoder; }
enum class TieredQIndexMode { fast, full };
enum class TieredQKernel { automatic, scalar, avx2, avx512, neon };
struct TieredQConfig {
  uint32_t dimensions = 512;
  uint32_t code_bits = 512;
  uint32_t num_centroids = 256;
  bool rotate_tiers = true;
  // Training-only cap on bits per coordinate; persisted models record only the resulting tiers.
  uint32_t max_bits = 8;
  // Training-only: only the top-N PCA ranks may exceed 6 bits (UINT32_MAX disables the restriction).
  uint32_t wide_dims = UINT32_MAX;
  // Training-only: the index mode whose storage layout bit allocation optimises for. Fast models with all tiers
  // <= 4 bits use the x86 nibble lookup, so their 3-bit tiers take an additive codebook whose codes split into
  // 2- and 1-bit nibble parts; they are kept only if padded storage fits code_bits and they beat 2/4-bit tiers.
  // Full mode packs codes densely and keeps the Lloyd-Max 3-bit codebook. Either model serves in either mode.
  TieredQIndexMode target_mode = TieredQIndexMode::fast;
  static TieredQConfig target_bytes(uint32_t dimensions, uint32_t bytes);
  static TieredQConfig target_code_bits(uint32_t dimensions, uint32_t bits);
  static TieredQConfig target_bytes_per_vector(uint32_t dimensions, uint32_t bytes) {
    return target_bytes(dimensions, bytes);
  }
  void validate() const;
};
struct TieredQTier {
  uint32_t start, dimensions, bits, scale_group, bit_offset; // persisted scale_group is reserved and must be zero
};
// One FP32 scan lookup over count whole coordinates starting at rank first (tieredq_scan_groups.h).
struct TieredQScanGroup {
  static constexpr uint32_t wide = UINT32_MAX;  // table value of a coordinate scored from its shared codebook
  uint32_t bit_offset, bits, first, count, table;
};
struct TieredQQuery {
  // Float LUT entries already include sigma. One table per coded coordinate.
  std::vector<float> lut, centroid_scores;
  // q.c / |c|^2 per centroid: scaling a row's centroid correction by its centroid's weight removes exactly the
  // error component along c, leaving (q - weight * c).(r - r_hat).
  std::vector<float> correction_weights;
  std::vector<size_t> offsets;
  // Generated coordinate weights let index scans reuse shared wide codebooks.
  // The public row scorer continues to consume lut directly.
  std::vector<float> weights;
  // FP32 grouped scan plan and its float tables; queries without them use the per-coordinate kernels.
  std::vector<TieredQScanGroup> scan_groups;
  std::vector<float> scan_tables;
};
struct TieredQHit { float score; uint32_t id; };
// Right-hand GEMM operand in K-major panels of PANEL columns (zero padded).
struct TieredQPanels {
  static constexpr size_t PANEL = 48;
  size_t depth = 0, columns = 0;
  std::vector<float> data;
  size_t padded_columns() const { return (columns + PANEL - 1) / PANEL * PANEL; }
};
class TieredQIndex;
class TieredQ {
 public:
  static TieredQ train(TieredQConfig config, const float* sample, size_t rows, uint64_t seed = 0);
  const TieredQConfig& config() const { return config_; }
  const std::vector<TieredQTier>& tiers() const { return tiers_; }
  size_t coded_dimensions() const { return sigma_.size(); }
  size_t code_bytes() const { return (size_t(config_.code_bits) + 7) / 8; }
  size_t metadata_bytes_per_vector() const;
  size_t bytes_per_vector() const { return code_bytes() + metadata_bytes_per_vector(); }
  size_t model_bytes() const;
  // Resident model memory: serving state, plus encoder state once encode() has built it.
  size_t memory_bytes() const;
  uint32_t model_id() const;
  void decode(const uint8_t* payload, size_t count, float* rows) const;
  // Plane buffers contain ceil(count/16) complete blocks of 16 lanes.
  // Automatic packed scoring selects NEON on AArch64; explicit NEON row scoring uses scalar.
  void pack(const uint8_t* rows, size_t count, uint8_t* planes) const;
  void unpack(const uint8_t* planes, size_t count, uint8_t* rows) const;
  // Scores always consume the correction embedded in each row.
  void score_planes(const uint8_t* planes, size_t count, const TieredQQuery& query,
                    float* scores, TieredQKernel kernel = TieredQKernel::automatic,
                    bool validate_query = true) const;
  // Batched score_planes: one pass over the plane block serves up to four queries.
  // Query i writes scores[i * score_stride + doc]; score_stride must be at least count.
  void score_planes_batch(const uint8_t* planes, size_t count, const TieredQQuery* queries,
                          size_t query_count, float* scores, size_t score_stride,
                          TieredQKernel kernel = TieredQKernel::automatic,
                          bool validate_query = true) const;
  // Rows are contiguous finite float32 vectors; scores estimate inner products.
  std::vector<uint8_t> encode(const float* rows, size_t count) const;
  // Writes count * bytes_per_vector() bytes; parallel over row blocks outside OpenMP regions.
  // Optional output receives the stored, rounded centroid correction for each row.
  // Metadata uses round-to-nearest-even binary16 and rejects overflow/nonfinite values.
  void encode(const float* rows, size_t count, uint8_t* out, float* corrections = nullptr) const;
  TieredQQuery query(const float* vector) const;
  // Projection and centroid terms without LUTs or scan groups, for weight-based fast scans.
  // Use query() to prepare input for score() and score_planes().
  TieredQQuery query_weights(const float* vector) const;
  // Batched query_weights over contiguous rows; results equal per-row calls exactly.
  std::vector<TieredQQuery> query_weights(const float* vectors, size_t rows) const;
  void score(const uint8_t* payload, size_t count, const TieredQQuery& query,
             float* scores, TieredQKernel kernel = TieredQKernel::automatic) const;
  // Format 3 stores the PCA basis as IEEE binary16; format-2 files (FP32 basis) still load and serve unchanged.
  static constexpr uint32_t model_format = 3;
  static constexpr uint32_t fp32_basis_model_format = 2;
  static constexpr size_t metadata_bytes = 5;
  static bool kernel_available(TieredQKernel kernel);
  void save(const std::string& path) const;
  static TieredQ load(const std::string& path);
  // Owned model bytes for embedding in an engine's storage container.
  std::vector<uint8_t> serialize() const;
  static TieredQ deserialize(const std::vector<uint8_t>& data);
 private:
  friend class TieredQIndex;
  friend struct tieredq_detail::FastQueryBasis;
  TieredQ() = default;
  TieredQConfig config_;
  std::vector<TieredQTier> tiers_;
  std::vector<float> centroids_, basis_, mean_, sigma_;
  // Binary16 basis of format-3 models; exactly one of basis_ and basis_half_ holds the d x m basis.
  std::vector<uint16_t> basis_half_;
  float basis_value(size_t index) const;
  std::vector<std::vector<float>> levels_;
  // Derived serving state (not serialized): float half norms supply query correction weights.
  std::vector<float> centroid_half_norms_;
  // Derived scan geometry (not serialized): window and shift per grouped FP32 lookup, built from tiers_.
  std::vector<tieredq_detail::ScanGroupPlan> scan_plan_;
  // Encoder-only state (fused panels, FP64 centroid projections), built on first encode().
  std::shared_ptr<tieredq_detail::LazyEncoder> encoder_;
  void prepare_serving();
  tieredq_detail::EncoderData build_encoder() const;
  void encode_block(const float* rows, size_t count, uint8_t* out, float* corrections, tieredq_detail::StageProfile* profile = nullptr) const;
  void encode_rows(const float* rows, size_t count, uint8_t* out, float* corrections) const;
  // Up to eight validated rows; shares basis traffic and vectorizes centroid dots across rows.
  void project_query_tile(const float* vectors, size_t rows, TieredQQuery* out) const;
  void validate_model() const;
};
class TieredQIndex {
 public:
  // Fast mode owns only its approximate search representation; the mode is persisted.
  explicit TieredQIndex(TieredQ codec, TieredQIndexMode mode = TieredQIndexMode::fast);
  ~TieredQIndex();
  TieredQIndex(const TieredQIndex&) = default;
  TieredQIndex& operator=(const TieredQIndex&) = default;
  TieredQIndex(TieredQIndex&& other) noexcept
      : codec_(std::move(other.codec_)), fast_mode_(other.fast_mode_), payload_(std::move(other.payload_)),
        fast_(std::move(other.fast_)), count_(std::exchange(other.count_,0)) {}
  TieredQIndex& operator=(TieredQIndex&& other) noexcept {
    if (this!=&other) {
      codec_=std::move(other.codec_); fast_mode_=other.fast_mode_; payload_=std::move(other.payload_);
      fast_=std::move(other.fast_); count_=std::exchange(other.count_,0);
    }
    return *this;
  }
  const TieredQ& codec() const { return codec_; }
  size_t size() const { return count_; }
  TieredQIndexMode mode() const { return fast_mode_ ? TieredQIndexMode::fast : TieredQIndexMode::full; }
  // Actual fast-data dispatch; false when no fast representation is present.
  bool uses_scalar_fast_scan() const;
  static bool native_fast_scan_available();
  size_t payload_bytes() const;
  // Allocated vector data and fast-cache state, including its derived query projection/codebooks.
  // Excludes codec memory, allocator bookkeeping, query state and top-k scratch.
  size_t storage_bytes() const;
  size_t bytes_per_vector() const;
  void add(const float* rows, size_t count);
  std::vector<TieredQHit> search(const float* query, size_t k,
                                TieredQKernel kernel = TieredQKernel::automatic) const;
  // Optional auxiliary cache for exact-mode indexes; fast-only indexes need no rebuild.
  // Models containing tiers wider than 4 bits preserve code IDs for FP32 scoring. Other models
  // use int8 levels on ARM and compact lookups on x86 when all tiers fit four bits.
  // Successful nonempty appends invalidate only an exact-mode auxiliary cache.
  // No supported native kernel leaves exact-mode indexes without this optional cache.
  void build_fast_scan();
  bool has_fast_scan() const { return fast_!=nullptr; }
  // Full scan using fast data, or exact fallback when an exact-mode cache is absent.
  std::vector<TieredQHit> search_fast(const float* query, size_t k) const;
  // Batched search over rows x dimensions queries. Results are rows x min(k, size()) in
  // row-major order; the caller owns both output arrays. Runs OpenMP workers internally.
  void search_batch(const float* queries, size_t rows, size_t k,
                    float* scores, uint32_t* ids) const;
  // Same contract as search_batch, using fast data. A portable scorer handles saved
  // representations unsupported by local native kernels; absent exact-mode caches fall back.
  void search_batch_fast(const float* queries, size_t rows, size_t k,
                         float* scores, uint32_t* ids) const;
  void save(const std::string& path) const;
  static TieredQIndex load(const std::string& path);
 private:
  std::shared_ptr<tieredq_detail::FastScanCache> make_fast_storage(
      const uint8_t* rows, size_t count, bool lookup, int level_max) const;
  std::shared_ptr<tieredq_detail::FastScanCache> make_fast_storage(const uint8_t* rows, size_t count) const;
  TieredQ codec_;
  bool fast_mode_ = false;
  // Exact-mode planes across 16 documents; empty for fast-only indexes.
  std::vector<uint8_t> payload_;
  std::shared_ptr<tieredq_detail::FastScanCache> fast_;
  size_t count_ = 0;
};
} // namespace dv
