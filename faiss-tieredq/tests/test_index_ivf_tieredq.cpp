// IndexIVFTieredQ against TieredQIVFIndex and Faiss's own index contract, for inner product, cosine and L2:
// - parity: with the same codec and partition (L2 quantizer on TieredQ's centroids) and equal nprobe, ids and scores
//   equal TieredQIVFIndex's up to tie order (L2: distances are the negated scores);
// - training reproduces the codec TieredQ's Python build trains on the same rows;
// - lists in blocks of their size rounded up to 16 documents; appends in small batches (growing blocks) search as one
//   bulk add;
// - write/read, remove_ids, merge_from, reconstruct (direct map), sa_encode/sa_decode and distance_to_code.
#include "faiss_tieredq/IndexIVFTieredQ.h"
#include <faiss/IndexFlat.h>
#include <faiss/impl/IDSelector.h>
#include <tieredq/tieredq.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <unistd.h>
#include <vector>

namespace {
using faiss::idx_t;
using faiss_tieredq::IndexIVFTieredQ;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }

constexpr size_t dimensions = 64, lists = 16, rows = 3000, queries = 20, k = 25, probes = 4;
constexpr uint32_t code_bits = 2 * dimensions;

std::vector<float> normal_rows(size_t count, uint32_t seed) {
  std::mt19937 random(seed); std::normal_distribution<float> normal;
  std::vector<float> data(count * dimensions);
  for (auto& value : data) value = normal(random);
  return data;
}

struct Hits { std::vector<float> distances; std::vector<idx_t> labels; };

Hits search(const faiss::Index& index, const std::vector<float>& query_rows, size_t nprobe) {
  Hits hits{std::vector<float>(queries * k), std::vector<idx_t>(queries * k)};
  faiss::SearchParametersIVF params; params.nprobe = nprobe;
  index.search(idx_t(queries), query_rows.data(), idx_t(k), hits.distances.data(), hits.labels.data(), &params);
  return hits;
}

// Same multiset of (id, score) per query, so ties may come in either order.
bool same_hits(const Hits& a, const Hits& b) {
  for (size_t q = 0; q < queries; ++q) {
    std::vector<std::pair<idx_t, float>> left, right;
    for (size_t i = 0; i < k; ++i) {
      left.emplace_back(a.labels[q * k + i], a.distances[q * k + i]);
      right.emplace_back(b.labels[q * k + i], b.distances[q * k + i]);
    }
    std::sort(left.begin(), left.end()); std::sort(right.begin(), right.end());
    if (left != right) return false;
  }
  return true;
}

// The k-th score may tie across documents, so the last kept ids can differ; every score agrees.
bool same_scores(const Hits& a, const Hits& b) { return a.distances == b.distances; }

std::unique_ptr<IndexIVFTieredQ> make_index(faiss::MetricType metric, bool cosine) {
  auto index = std::make_unique<IndexIVFTieredQ>(new faiss::IndexFlatL2(dimensions), dimensions, lists, code_bits, metric, cosine);
  index->own_fields = true;
  return index;
}

std::vector<float> normalized(std::vector<float> rows) {
  for (size_t row = 0; row < rows.size() / dimensions; ++row) {
    double square = 0;
    for (size_t j = 0; j < dimensions; ++j) square += double(rows[row * dimensions + j]) * rows[row * dimensions + j];
    for (size_t j = 0; j < dimensions; ++j) rows[row * dimensions + j] = float(rows[row * dimensions + j] / std::sqrt(square));
  }
  return rows;
}

// Cosine ranks TieredQ's lists on normalized vectors, so the quantizer gets them normalized too (Faiss's convention).
void check_parity(tq::TieredQMetric metric, const std::vector<float>& raw_data, const std::vector<float>& raw_queries) {
  const bool l2 = metric == tq::TieredQMetric::l2, cosine = metric == tq::TieredQMetric::cosine;
  const auto data = cosine ? normalized(raw_data) : raw_data, query_rows = cosine ? normalized(raw_queries) : raw_queries;
  const auto faiss_metric = l2 ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
  // Training: the plugin's codec is the one TieredQ's Python adapter trains on the same rows.
  auto trained = make_index(faiss_metric, cosine);
  trained->train(idx_t(rows), data.data());
  // TieredQIVFIndex on the trained codec with the plugin quantizer's centroids as its lists.
  tq::TieredQIVFIndex reference(trained->codec(), lists);
  std::vector<float> centroids(lists * dimensions);
  trained->quantizer->reconstruct_n(0, idx_t(lists), centroids.data());
  reference.set_lists(centroids.data());
  reference.build(data.data(), rows);
  trained->add(idx_t(rows), data.data());
  for (size_t nprobe : {probes, lists}) {
    Hits expected{std::vector<float>(queries * k), std::vector<idx_t>(queries * k)};
    for (size_t q = 0; q < queries; ++q) {
      const auto hits = reference.search(query_rows.data() + q * dimensions, k, nprobe);
      for (size_t i = 0; i < k; ++i) {
        expected.labels[q * k + i] = i < hits.size() ? idx_t(hits[i].id) : -1;
        expected.distances[q * k + i] = i < hits.size() ? (l2 ? -hits[i].score : hits[i].score) : 0;
      }
    }
    const auto found = search(*trained, query_rows, nprobe);
    require(same_scores(found, expected), "IndexIVFTieredQ scores differ from TieredQIVFIndex");
    require(same_hits(found, expected) || nprobe == probes, "IndexIVFTieredQ hits differ from TieredQIVFIndex at full probe");
  }
}

void check_contract(faiss::MetricType metric, bool cosine, const std::vector<float>& data, const std::vector<float>& query_rows) {
  const auto path = (std::filesystem::temp_directory_path() / ("faiss_tieredq_" + std::to_string(getpid()) + ".tq")).string();
  auto index = make_index(metric, cosine);
  index->train(idx_t(rows), data.data());
  std::vector<idx_t> ids(rows);
  for (size_t i = 0; i < rows; ++i) ids[i] = idx_t(1000 + 7 * i);
  index->add_with_ids(idx_t(rows), data.data(), ids.data());
  const auto before = search(*index, query_rows, probes);
  // A bulk add lays each list out for its size rounded up to 16 documents.
  for (size_t list = 0; list < lists; ++list) {
    const size_t size = index->lists().list_size(list), capacity = index->lists().capacity(list);
    require(capacity >= size && capacity < size + faiss_tieredq::TieredQInvertedLists::capacity_granularity &&
            capacity % faiss_tieredq::TieredQInvertedLists::capacity_granularity == 0, "bulk add left slack capacity");
  }
  // Appends of 7 rows grow the blocks many times and search as the bulk add.
  auto grown = make_index(metric, cosine);
  grown->quantizer->add(idx_t(lists), dynamic_cast<faiss::IndexFlat*>(index->quantizer)->get_xb());
  grown->set_codec(index->codec());
  for (size_t first = 0; first < rows; first += 7) {
    const size_t count = std::min<size_t>(7, rows - first);
    grown->add_with_ids(idx_t(count), data.data() + first * dimensions, ids.data() + first);
  }
  require(same_hits(search(*grown, query_rows, probes), before), "appended index searches differently from the bulk add");
  // Persistence.
  faiss_tieredq::write_index_tieredq(*index, path);
  std::unique_ptr<IndexIVFTieredQ> loaded(faiss_tieredq::read_index_tieredq(path));
  std::filesystem::remove(path);
  require(loaded->ntotal == index->ntotal && same_hits(search(*loaded, query_rows, probes), before), "read index searches differently");
  // Two halves merged equal the whole.
  auto first = make_index(metric, cosine), second = make_index(metric, cosine);
  for (auto* half : {first.get(), second.get()}) {
    half->quantizer->add(idx_t(lists), dynamic_cast<faiss::IndexFlat*>(index->quantizer)->get_xb());
    half->set_codec(index->codec());
  }
  first->add_with_ids(idx_t(rows / 2), data.data(), ids.data());
  second->add_with_ids(idx_t(rows - rows / 2), data.data() + rows / 2 * dimensions, ids.data() + rows / 2);
  first->merge_from(*second, 0);
  require(first->ntotal == idx_t(rows) && second->ntotal == 0, "merge_from counts");
  require(same_scores(search(*first, query_rows, probes), before), "merged index searches differently");
  // Removing every third id: the rest keep their scores, the removed never return.
  std::vector<idx_t> removed;
  for (size_t i = 0; i < rows; i += 3) removed.push_back(ids[i]);
  faiss::IDSelectorBatch selector(removed.size(), removed.data());
  require(loaded->remove_ids(selector) == removed.size(), "remove_ids count");
  const auto after = search(*loaded, query_rows, probes);
  const auto scan = [&](const Hits& hits) {
    std::map<std::pair<size_t, idx_t>, float> kept;
    for (size_t i = 0; i < hits.labels.size(); ++i) kept[{i / k, hits.labels[i]}] = hits.distances[i];
    return kept;
  };
  const auto old_scores = scan(before);
  for (const auto& [key, distance] : scan(after)) {
    const idx_t id = key.second;
    require(id < 0 || !selector.is_member(id), "removed id returned");
    if (old_scores.count(key)) require(old_scores.at(key) == distance, "remove_ids changed a kept row's score");
  }
  // Reconstruct through a direct map, and the standalone codec, both decode the codec's rows.
  auto mapped = make_index(metric, cosine);
  mapped->quantizer->add(idx_t(lists), dynamic_cast<faiss::IndexFlat*>(index->quantizer)->get_xb());
  mapped->set_codec(index->codec());
  mapped->make_direct_map(true);
  mapped->add(idx_t(rows), data.data());
  std::vector<float> rebuilt(dimensions), decoded(rows * dimensions);
  index->codec().decode(index->codec().encode(data.data(), rows).data(), rows, decoded.data());
  for (size_t row : {size_t(0), rows / 2, rows - 1}) {
    mapped->reconstruct(idx_t(row), rebuilt.data());
    require(std::equal(rebuilt.begin(), rebuilt.end(), decoded.begin() + row * dimensions), "reconstruct differs from decode");
  }
  std::vector<uint8_t> standalone(rows * index->sa_code_size());
  index->sa_encode(idx_t(rows), data.data(), standalone.data());
  std::vector<float> standalone_decoded(rows * dimensions);
  index->sa_decode(idx_t(rows), standalone.data(), standalone_decoded.data());
  require(standalone_decoded == decoded, "sa_decode differs from decode");
  // distance_to_code scores one row-major code exactly; the scan's fast estimate lies close to it.
  std::unique_ptr<faiss::InvertedListScanner> scanner(index->get_InvertedListScanner());
  scanner->set_query(query_rows.data());
  const auto codes = index->codec().encode(data.data(), rows);
  std::vector<float> exact(rows);
  for (size_t row = 0; row < rows; ++row) exact[row] = scanner->distance_to_code(codes.data() + row * index->code_size);
  Hits full{std::vector<float>(k), std::vector<idx_t>(k)};
  faiss::SearchParametersIVF all; all.nprobe = lists;
  index->search(1, query_rows.data(), idx_t(k), full.distances.data(), full.labels.data(), &all);
  for (size_t i = 0; i < k; ++i) {
    const float reference = exact[size_t(std::find(ids.begin(), ids.end(), full.labels[i]) - ids.begin())];
    require(std::fabs(full.distances[i] - reference) <= 0.05f * (1 + std::fabs(reference)), "scan drifts from distance_to_code");
  }
}
}  // namespace

int main() {
  try {
    const auto data = normal_rows(rows, 1), query_rows = normal_rows(queries, 2);
    for (auto metric : {tq::TieredQMetric::inner_product, tq::TieredQMetric::cosine, tq::TieredQMetric::l2})
      check_parity(metric, data, query_rows);
    check_contract(faiss::METRIC_INNER_PRODUCT, false, data, query_rows);
    check_contract(faiss::METRIC_INNER_PRODUCT, true, data, query_rows);
    check_contract(faiss::METRIC_L2, false, data, query_rows);
    std::puts("OK");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
