// The InvertedListScanner of IndexIVFTieredQ: one query's TieredQ fast-scan tables, applied to a list's block in one call
// of the fast kernels, with scores mapped to Faiss's convention (larger-is-better similarities for inner product, squared
// distances for L2). Mirrors Faiss's run_scan_codes1: the selector skips results, the handler's threshold gates them.
// With by_residual each list's scores add the query's inner product with its centroid, from set_list's coarse distance.
#pragma once
#include "faiss_tieredq/IndexIVFTieredQ.h"
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/invlists/DirectMap.h>
#include <numeric>
#include <optional>
#include <vector>

namespace faiss_tieredq {
class ListScanner final : public faiss::InvertedListScanner {
 public:
  ListScanner(const IndexIVFTieredQ& index, std::shared_ptr<ScannerPool> pool, bool store_pairs_in, const faiss::IDSelector* sel_in)
      : InvertedListScanner(store_pairs_in, sel_in), pool_(std::move(pool)), blocks_(pool_->blocks()), lists_(index.lists()),
        dimensions_(index.d),
        distances_(index.metric_type == faiss::METRIC_L2),
        centroid_norms_(index.by_residual ? &index.centroid_norms : nullptr),
        coarse_inner_product_(index.quantizer->metric_type == faiss::METRIC_INNER_PRODUCT) {
    // A sorted id range lets Faiss offset the codes pointer by rows, which blocks do not allow.
    if (const auto* range = dynamic_cast<const faiss::IDSelectorRange*>(sel_in))
      FAISS_THROW_IF_NOT_MSG(!range->assume_sorted, "IndexIVFTieredQ does not support sorted IDSelectorRange");
    keep_max = !distances_;
    code_size = index.code_size;
  }

  ~ListScanner() override { if (scanner_) pool_->give(std::move(*scanner_)); }

  void set_query(const float* query) override {
    query_.assign(query, query + dimensions_);
    query_norm_ = std::inner_product(query_.begin(), query_.end(), query_.begin(), 0.0f);
    if (scanner_) blocks_.reset(*scanner_, query);
    else scanner_.emplace(pool_->take(query));
    single_.reset();
  }

  void set_list(idx_t list, float coarse_dis) override {
    list_no = list;
    if (!centroid_norms_) return;
    bias_ = coarse_inner_product_ ? coarse_dis : 0.5f * (query_norm_ + (*centroid_norms_)[size_t(list)] - coarse_dis);
  }

  // One row-major encode() code, scored exactly by the codec (refinement, range checks); not the scan path.
  float distance_to_code(const uint8_t* code) const override {
    if (!single_) single_.emplace(blocks_.codec().query(query_.data()));
    float score = 0;
    blocks_.codec().score(code, 1, *single_, &score);
    to_metric(&score, 1);
    return score;
  }

  // codes: the list's block (get_codes), laid out for its capacity; list_size may stop short of the list's rows.
  size_t scan_codes(size_t list_size, const uint8_t* codes, const idx_t* ids, faiss::ResultHandler& handler) const override {
    if (!list_size) return 0;
    scores_.resize(list_size);
    scanner_->score(codes, lists_.capacity(size_t(list_no)), list_size, scores_.data());
    to_metric(scores_.data(), list_size);
    handler.stats.scan_cnt += list_size;
    size_t updates = 0;
    float threshold = handler.threshold;
    for (size_t j = 0; j < list_size; ++j) {
      const idx_t id = store_pairs ? idx_t(faiss::lo_build(list_no, j)) : ids[j];
      if (sel && !sel->is_member(id)) continue;
      const float dis = scores_[j];
      if (!(keep_max ? dis > threshold : dis < threshold)) continue;
      if (handler.add_result(dis, id)) {
        ++handler.stats.nheap_updates; ++updates;
        threshold = handler.threshold;
      }
    }
    return updates;
  }

 private:
  // TieredQ's metric scores (negative squared distances for L2), negated into Faiss distances for L2.
  void to_metric(float* scores, size_t count) const {
    blocks_.codec().map_scores(query_.data(), scores, count);
    if (distances_) for (size_t i = 0; i < count; ++i) scores[i] = -scores[i];
    if (centroid_norms_) for (size_t i = 0; i < count; ++i) scores[i] += bias_;
  }

  std::shared_ptr<ScannerPool> pool_;
  const tq::TieredQFastBlocks& blocks_;
  const TieredQInvertedLists& lists_;
  size_t dimensions_;
  bool distances_;
  const std::vector<float>* centroid_norms_;
  bool coarse_inner_product_;
  float query_norm_ = 0, bias_ = 0;
  std::vector<float> query_;
  mutable std::optional<tq::TieredQFastBlocks::Scanner> scanner_;
  mutable std::optional<tq::TieredQQuery> single_;
  mutable std::vector<float> scores_;
};
}  // namespace faiss_tieredq
