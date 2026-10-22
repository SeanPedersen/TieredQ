// Runtime-sized TieredQ. Link dv_tieredq (LAPACK/BLAS); no Python dependency.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <utility>
#include "tieredq/detail/tieredq_scan_plan.h"

namespace dv {
namespace tieredq_detail { class StageProfile; }
enum class TieredQKernel { automatic, scalar, avx2, avx512 };
struct TieredQConfig {
  uint32_t dimensions = 512;
  uint32_t code_bits = 512;
  uint32_t num_centroids = 256;
  bool rotate_tiers = true;
  // Training-only cap on bits per coordinate; persisted models record only the resulting tiers.
  uint32_t max_bits = 8;
  // Training-only: only the top-N PCA ranks may exceed 6 bits (UINT32_MAX disables the restriction).
  uint32_t wide_dims = UINT32_MAX;
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
  size_t memory_bytes() const;
  uint32_t model_id() const;
  void decode(const uint8_t* payload, size_t count, float* rows) const;
  // Plane buffers contain ceil(count/16) complete blocks of 16 lanes.
  void pack(const uint8_t* rows, size_t count, uint8_t* planes) const;
  void unpack(const uint8_t* planes, size_t count, uint8_t* rows) const;
  // Scores always consume the correction embedded in each row.
  void score_planes(const uint8_t* planes, size_t count, const TieredQQuery& query,
                    float* scores, TieredQKernel kernel = TieredQKernel::automatic,
                    bool validate_query = true) const;
  // Rows are contiguous finite float32 vectors; scores estimate inner products.
  std::vector<uint8_t> encode(const float* rows, size_t count) const;
  // Writes count * bytes_per_vector() bytes; parallel over row blocks outside OpenMP regions.
  // Optional output receives the stored, rounded centroid correction for each row.
  // Metadata uses round-to-nearest-even binary16 and rejects overflow/nonfinite values.
  void encode(const float* rows, size_t count, uint8_t* out, float* corrections = nullptr) const;
  TieredQQuery query(const float* vector) const;
  void score(const uint8_t* payload, size_t count, const TieredQQuery& query,
             float* scores, TieredQKernel kernel = TieredQKernel::automatic) const;
  static constexpr uint32_t model_format = 2;
  static constexpr size_t metadata_bytes = 5;
  static bool kernel_available(TieredQKernel kernel);
  void save(const std::string& path) const;
  static TieredQ load(const std::string& path);
 private:
  friend class TieredQIndex;
  TieredQ() = default;
  TieredQConfig config_;
  std::vector<TieredQTier> tiers_;
  std::vector<float> centroids_, basis_, mean_, sigma_;
  std::vector<std::vector<float>> levels_;
  // Derived state (not serialized): fused FP32 panels and exact centroid projections.
  // Float half norms also supply query correction weights.
  TieredQPanels encode_panels_;
  std::vector<float> centroid_half_norms_;
  // retains FP64 centroid projections and half norms for residuals and correction.
  std::vector<double> centroid_projections_, exact_centroid_half_norms_;
  // Derived scan geometry (not serialized): window and shift per grouped FP32 lookup, built from tiers_.
  std::vector<tieredq_detail::ScanGroupPlan> scan_plan_;
  void prepare_encoder();
  void encode_block(const float* rows, size_t count, uint8_t* out, float* corrections, tieredq_detail::StageProfile* profile = nullptr) const;
  void encode_rows(const float* rows, size_t count, uint8_t* out, float* corrections) const;
  void validate_model() const;
  std::vector<uint8_t> serialize() const;
  static TieredQ deserialize(const std::vector<uint8_t>& data);
};
class TieredQIndex {
 public:
  explicit TieredQIndex(TieredQ codec) : codec_(std::move(codec)) {}
  TieredQIndex(const TieredQIndex&) = default;
  TieredQIndex& operator=(const TieredQIndex&) = default;
  TieredQIndex(TieredQIndex&& other) noexcept
      : codec_(std::move(other.codec_)), payload_(std::move(other.payload_)),
        count_(std::exchange(other.count_,0)) {}
  TieredQIndex& operator=(TieredQIndex&& other) noexcept {
    if (this!=&other) {
      codec_=std::move(other.codec_); payload_=std::move(other.payload_);
      count_=std::exchange(other.count_,0);
    }
    return *this;
  }
  const TieredQ& codec() const { return codec_; }
  size_t size() const { return count_; }
  size_t payload_bytes() const { return count_ * codec_.bytes_per_vector(); }
  void add(const float* rows, size_t count);
  std::vector<TieredQHit> search(const float* query, size_t k,
                                TieredQKernel kernel = TieredQKernel::automatic) const;
  void save(const std::string& path) const;
  static TieredQIndex load(const std::string& path);
 private:
  TieredQ codec_;
  // Packed byte planes across 16 documents; only the last block is padded.
  std::vector<uint8_t> payload_;
  size_t count_ = 0;
};
} // namespace dv
