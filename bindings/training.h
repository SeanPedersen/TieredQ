// Codec configuration and the seeded training sample shared by the flat, IVF and nested adapters: every index trains
// on the same rows of its first training batch, so a nested refinement can retrain on exactly its prefix's sample.
#pragma once
#include <tieredq/tieredq.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace tieredq_python {
// A training sample smaller than twice the cap cannot seed that many
// centroids, so the default shrinks to fit instead of failing at train time.
constexpr uint32_t maximum_auto_centroids = 256;
constexpr uint64_t training_seed = 0;
// Recall saturates at 8-16K training rows on 1M-row sets (#19.2), so larger
// batches train on this many rows.
constexpr size_t training_sample_rows = 32768;
constexpr size_t minimum_training_rows = 2;

inline uint32_t auto_centroids(size_t sample_rows) {
  return std::min(maximum_auto_centroids, static_cast<uint32_t>(sample_rows / 2));
}
inline dv::TieredQMetric parse_metric(const std::string& metric) {
  if (metric=="ip") return dv::TieredQMetric::inner_product;
  if (metric=="cosine") return dv::TieredQMetric::cosine;
  if (metric=="l2") return dv::TieredQMetric::l2;
  throw std::invalid_argument("metric must be 'ip', 'cosine' or 'l2'");
}
inline dv::TieredQFineTiers parse_fine_tiers(const std::string& mode) {
  if (mode=="auto") return dv::TieredQFineTiers::automatic;
  if (mode=="off") return dv::TieredQFineTiers::off;
  if (mode=="on") return dv::TieredQFineTiers::on;
  throw std::invalid_argument("fine_tiers must be 'auto', 'off' or 'on'");
}
inline const char* metric_name(dv::TieredQMetric metric) {
  return metric==dv::TieredQMetric::l2 ? "l2" : metric==dv::TieredQMetric::cosine ? "cosine" : "ip";
}
// Rows of a seeded sample of at most training_sample_rows of count rows; all of them, in order, when fewer.
inline std::vector<uint32_t> training_sample_ids(size_t count) {
  std::vector<uint32_t> ids(count);
  std::iota(ids.begin(), ids.end(), 0);
  if (count > training_sample_rows) std::shuffle(ids.begin(), ids.end(), std::mt19937_64(training_seed));
  ids.resize(std::min(count, training_sample_rows));
  return ids;
}
// Trains config on the training sample of rows (count x dimensions), read in place.
inline dv::TieredQ train_codec(dv::TieredQConfig config, const float* rows, size_t count) {
  if (count < minimum_training_rows) throw std::runtime_error("training needs at least 2 rows");
  const auto ids = training_sample_ids(count);
  config.num_centroids = auto_centroids(ids.size());
  return dv::TieredQ::train_sampled(config, rows, count, ids.data(), ids.size(), training_seed);
}
// The training sample of rows gathered into one array, for trainers that read contiguous rows.
inline std::vector<float> training_sample(const float* rows, size_t count, size_t width) {
  const auto ids = training_sample_ids(count);
  std::vector<float> sample(ids.size() * width);
  for (size_t i = 0; i < ids.size(); ++i)
    std::copy_n(rows + size_t(ids[i]) * width, width, sample.data() + i * width);
  return sample;
}
}  // namespace tieredq_python
