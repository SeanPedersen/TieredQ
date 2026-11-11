// Two-level compressed coarse router of TieredQIVFIndex (#47): an L2 TieredQ IVF index whose rows are the nonempty
// lists' centroids and whose coarse lists are the supers, plus a row-major binary16 copy of those centroids for the
// exact re-rank of the router's head. Single-query routing then reads the supers and a few thousand compressed
// centroids instead of streaming every centroid (about 9 ms per query at 64K x 1024).
#pragma once
#include "tieredq/tieredq.h"
#include "tieredq_ivf_coarse.h"

namespace tq::tieredq_detail {
class CoarseRouter {
 public:
  // Routes the nonempty lists of coarse (list_rows[list] > 0).
  static CoarseRouter build(const CoarseQuantizer& coarse, const std::vector<uint32_t>& list_rows,
                            const TieredQRouterConfig& config);
  // The nprobe nearest routed lists to a coarse-space query, nearest first (fewer when the probed supers hold fewer
  // lists). super_probes 0 picks the config's automatic count.
  std::vector<uint32_t> probe_order(const CoarseQuantizer& coarse, const float* query, size_t nprobe,
                                    uint32_t super_probes) const;
  size_t automatic_super_probes(size_t nprobe) const;
  size_t allocated_bytes() const;
  std::vector<uint8_t> serialize() const;
  // Bytes written by serialize() for the same coarse quantizer, validated against the index's lists.
  static CoarseRouter deserialize(const CoarseQuantizer& coarse, const std::vector<uint32_t>& list_rows,
                                  const TieredQRouterConfig& config, const std::vector<uint8_t>& bytes);
  static void write_config(Writer& writer, const TieredQRouterConfig& config);
  template<class R> static TieredQRouterConfig read_config(R& reader);

 private:
  CoarseRouter(const CoarseQuantizer& coarse, TieredQRouterConfig config, TieredQIVFIndex members,
               std::vector<uint32_t> lists);
  TieredQRouterConfig config_;
  TieredQIVFIndex members_;     // member row r is list lists_[r]'s centroid; its coarse lists are the supers
  std::vector<uint32_t> lists_;
  std::vector<uint16_t> halves_; // member rows' centroids over their scale, row-major, as the coarse panels hold them
  size_t dimensions_;
};

template<class R> TieredQRouterConfig CoarseRouter::read_config(R& reader) {
  TieredQRouterConfig config;
  config.lists_per_super=reader.u32();
  config.bits_per_dimension=std::bit_cast<double>(reader.u64());
  config.rerank=reader.u32();
  config.expansion=std::bit_cast<double>(reader.u64());
  config.min_super_probes=reader.u32();
  config.seed=reader.u64();
  config.validate();
  return config;
}
} // namespace tq::tieredq_detail
