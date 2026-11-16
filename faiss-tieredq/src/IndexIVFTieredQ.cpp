// IndexIVFTieredQ: codec training, encoding, TieredQ block lists and their maintenance.
#include "faiss_tieredq/IndexIVFTieredQ.h"
#include "list_scanner.h"
#include <faiss/impl/FaissAssert.h>
#include <faiss/invlists/DirectMap.h>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace faiss_tieredq {
namespace {
// TieredQ's Python adapters train the codec on this seeded sample of the first training batch (bindings/training.h);
// the same rows give the same codec as a TieredQIVFIndex built from them.
constexpr size_t training_sample_rows = 32768;
constexpr uint64_t training_seed = 0;
constexpr uint32_t maximum_centroids = 256;
// Rows assigned and encoded per step of add(), bounding the assignment's scratch; codes of the whole add are kept
// so each list grows once.
constexpr idx_t assign_chunk_rows = idx_t(1) << 20;
// Lists past capacity grow by half (amortized appends); lists shrunk below half their capacity give it back.
constexpr size_t growth_divisor = 2, shrink_divisor = 2;

size_t round_up(size_t value, size_t multiple) { return (value + multiple - 1) / multiple * multiple; }
}  // namespace

std::vector<uint32_t> codec_training_sample(size_t count) {
  std::vector<uint32_t> ids(count);
  std::iota(ids.begin(), ids.end(), 0u);
  if (count > training_sample_rows) std::shuffle(ids.begin(), ids.end(), std::mt19937_64(training_seed));
  ids.resize(std::min(count, training_sample_rows));
  return ids;
}

TieredQInvertedLists::TieredQInvertedLists(size_t nlist_in, std::shared_ptr<const tq::TieredQFastBlocks> blocks_in)
    : InvertedLists(nlist_in, blocks_in->codec().bytes_per_vector()), blocks(std::move(blocks_in)), lists(nlist_in) {}

size_t TieredQInvertedLists::list_size(size_t list_no) const {
  return wide_ids_ ? lists[list_no].wide_ids.size() : lists[list_no].narrow_ids.size();
}

const idx_t* TieredQInvertedLists::get_ids(size_t list_no) const {
  const auto& list = lists[list_no];
  if (wide_ids_) return list.wide_ids.data();
  auto* ids = new idx_t[list.narrow_ids.size()];
  std::copy(list.narrow_ids.begin(), list.narrow_ids.end(), ids);
  return ids;
}

void TieredQInvertedLists::release_ids(size_t, const idx_t* ids) const {
  if (!wide_ids_) delete[] ids;
}

idx_t TieredQInvertedLists::get_single_id(size_t list_no, size_t offset) const {
  FAISS_THROW_IF_NOT(offset < list_size(list_no));
  return wide_ids_ ? lists[list_no].wide_ids[offset] : idx_t(lists[list_no].narrow_ids[offset]);
}

void TieredQInvertedLists::admit_ids(const idx_t* ids, size_t n) {
  if (wide_ids_ || std::all_of(ids, ids + n, [](idx_t id) { return id >= 0 && id <= idx_t(UINT32_MAX); })) return;
  use_wide_ids();
}

void TieredQInvertedLists::use_wide_ids() {
  if (wide_ids_) return;
  for (auto& list : lists) {
    list.wide_ids.assign(list.narrow_ids.begin(), list.narrow_ids.end());
    std::vector<uint32_t>().swap(list.narrow_ids);
  }
  wide_ids_ = true;
}

std::vector<idx_t> TieredQInvertedLists::ids(size_t list_no) const {
  const auto& list = lists[list_no];
  return wide_ids_ ? list.wide_ids : std::vector<idx_t>(list.narrow_ids.begin(), list.narrow_ids.end());
}

void TieredQInvertedLists::append_ids(size_t list_no, const idx_t* ids, size_t n) {
  auto& list = lists[list_no];
  if (wide_ids_) list.wide_ids.insert(list.wide_ids.end(), ids, ids + n);
  else for (size_t i = 0; i < n; ++i) list.narrow_ids.push_back(uint32_t(ids[i]));
}

void TieredQInvertedLists::set_ids(size_t list_no, size_t offset, const idx_t* ids, size_t n) {
  auto& list = lists[list_no];
  if (wide_ids_) std::copy_n(ids, n, list.wide_ids.begin() + offset);
  else for (size_t i = 0; i < n; ++i) list.narrow_ids[offset + i] = uint32_t(ids[i]);
}

void TieredQInvertedLists::reserve_ids(size_t list_no, size_t n) {
  if (wide_ids_) lists[list_no].wide_ids.reserve(n);
  else lists[list_no].narrow_ids.reserve(n);
}

const uint8_t* TieredQInvertedLists::get_single_code(size_t list_no, size_t offset) const {
  FAISS_THROW_IF_NOT(offset < list_size(list_no));
  auto* row = new uint8_t[code_size];
  const uint32_t slot = uint32_t(offset);
  blocks->rows(lists[list_no].block.data(), lists[list_no].capacity, &slot, 1, row);
  return row;
}

void TieredQInvertedLists::release_codes(size_t list_no, const uint8_t* codes) const {
  if (codes != lists[list_no].block.data()) delete[] codes;
}

std::vector<uint8_t> TieredQInvertedLists::rows(size_t list_no) const {
  const auto& list = lists[list_no];
  const size_t size = list_size(list_no);
  std::vector<uint8_t> out(size * code_size);
  if (!size) return out;
  std::vector<uint32_t> slots(size);
  std::iota(slots.begin(), slots.end(), 0u);
  blocks->rows(list.block.data(), list.capacity, slots.data(), slots.size(), out.data());
  return out;
}

void TieredQInvertedLists::set_capacity(size_t list_no, size_t rows_wanted) {
  auto& list = lists[list_no];
  const size_t capacity = round_up(rows_wanted, capacity_granularity), size = list_size(list_no);
  if (capacity == list.capacity) return;
  FAISS_THROW_IF_NOT(capacity >= size);
  // The layout depends on the capacity, so live rows move by decoding and rewriting them.
  const auto kept = rows(list_no);
  faiss::AlignedTableTightAlloc<uint8_t, tq::TieredQFastBlocks::alignment> block(blocks->block_bytes(capacity));
  std::memset(block.data(), 0, block.nbytes());
  for (size_t slot = 0; slot < size; ++slot) blocks->write_row(block.data(), capacity, slot, kept.data() + slot * code_size);
  std::swap(list.block.ptr, block.ptr);
  std::swap(list.block.numel, block.numel);
  list.capacity = capacity;
}

// Faiss calls add_entries serially; update_entries runs per list in parallel (DirectMap::remove_ids) with ids
// already stored, so admitting them never switches the width there.
size_t TieredQInvertedLists::add_entries(size_t list_no, size_t n_entry, const idx_t* ids, const uint8_t* codes) {
  admit_ids(ids, n_entry);
  auto& list = lists[list_no];
  const size_t offset = list_size(list_no);
  if (offset + n_entry > list.capacity) set_capacity(list_no, std::max(offset + n_entry, list.capacity + list.capacity / growth_divisor));
  append_ids(list_no, ids, n_entry);
  for (size_t i = 0; i < n_entry; ++i) blocks->write_row(list.block.data(), list.capacity, offset + i, codes + i * code_size);
  return offset;
}

void TieredQInvertedLists::update_entries(size_t list_no, size_t offset, size_t n_entry, const idx_t* ids,
                                          const uint8_t* codes) {
  FAISS_THROW_IF_NOT(offset + n_entry <= list_size(list_no));
  admit_ids(ids, n_entry);
  auto& list = lists[list_no];
  set_ids(list_no, offset, ids, n_entry);
  for (size_t i = 0; i < n_entry; ++i) blocks->write_row(list.block.data(), list.capacity, offset + i, codes + i * code_size);
}

// Slots past the new size keep stale rows, which scans of the first list_size() slots skip.
void TieredQInvertedLists::resize(size_t list_no, size_t new_size) {
  auto& list = lists[list_no];
  if (new_size > list.capacity) set_capacity(list_no, new_size);
  if (wide_ids_) list.wide_ids.resize(new_size);
  else list.narrow_ids.resize(new_size);
  if (new_size < list.capacity / shrink_divisor) set_capacity(list_no, new_size);
}

size_t TieredQInvertedLists::allocated_bytes() const {
  size_t bytes = 0;
  for (const auto& list : lists)
    bytes += list.block.nbytes() + list.narrow_ids.capacity() * sizeof(uint32_t) + list.wide_ids.capacity() * sizeof(idx_t);
  return bytes;
}

tq::TieredQFastBlocks::Scanner ScannerPool::take(const float* query) {
  {
    std::lock_guard lock(mutex_);
    if (!free_.empty()) {
      auto scanner = std::move(free_.back());
      free_.pop_back();
      blocks_->reset(scanner, query);
      return scanner;
    }
  }
  return blocks_->scanner(query);
}

void ScannerPool::give(tq::TieredQFastBlocks::Scanner scanner) {
  std::lock_guard lock(mutex_);
  free_.push_back(std::move(scanner));
}

IndexIVFTieredQ::IndexIVFTieredQ(faiss::Index* quantizer_in, size_t d_in, size_t nlist_in, uint32_t code_bits_in,
                                 faiss::MetricType metric, bool cosine_in)
    : IndexIVF(quantizer_in, d_in, nlist_in, 0, metric), code_bits(code_bits_in), cosine(cosine_in) {
  FAISS_THROW_IF_NOT_MSG(metric == faiss::METRIC_INNER_PRODUCT || metric == faiss::METRIC_L2,
                         "IndexIVFTieredQ supports inner product and L2");
  FAISS_THROW_IF_NOT_MSG(!cosine || metric == faiss::METRIC_INNER_PRODUCT, "cosine needs METRIC_INNER_PRODUCT");
  by_residual = false;
  is_trained = false;  // the codec trains in train_encoder (or comes from set_codec)
}

IndexIVFTieredQ::IndexIVFTieredQ() { by_residual = false; }

const tq::TieredQFastBlocks& IndexIVFTieredQ::blocks() const {
  FAISS_THROW_IF_NOT_MSG(blocks_, "IndexIVFTieredQ is not trained");
  return *blocks_;
}

const tq::TieredQ& IndexIVFTieredQ::codec() const { return blocks().codec(); }

TieredQInvertedLists& IndexIVFTieredQ::lists() const {
  auto* lists = dynamic_cast<TieredQInvertedLists*>(invlists);
  FAISS_THROW_IF_NOT_MSG(lists, "IndexIVFTieredQ needs its TieredQInvertedLists");
  return *lists;
}

void IndexIVFTieredQ::install(std::shared_ptr<const tq::TieredQFastBlocks> blocks) {
  FAISS_THROW_IF_NOT_MSG(ntotal == 0, "the codec of a nonempty IndexIVFTieredQ cannot change");
  FAISS_THROW_IF_NOT_MSG(idx_t(blocks->codec().config().dimensions) == d, "codec dimensions differ from the index's");
  blocks_ = std::move(blocks);
  scanners_ = std::make_shared<ScannerPool>(blocks_);
  code_size = blocks_->codec().bytes_per_vector();
  replace_invlists(new TieredQInvertedLists(nlist, blocks_), true);
}

void IndexIVFTieredQ::set_codec(const tq::TieredQ& codec, uint32_t format) {
  install(format ? std::make_shared<const tq::TieredQFastBlocks>(codec, format)
                 : std::make_shared<const tq::TieredQFastBlocks>(codec));
  is_trained = quantizer->is_trained && quantizer->ntotal == idx_t(nlist);
}

void IndexIVFTieredQ::train_encoder(idx_t n, const float* x, const idx_t*) {
  auto config = tq::TieredQConfig::target_code_bits(uint32_t(d), code_bits);
  config.metric = cosine ? tq::TieredQMetric::cosine
                         : metric_type == faiss::METRIC_L2 ? tq::TieredQMetric::l2 : tq::TieredQMetric::inner_product;
  config.fine_tiers = false;  // fast blocks scan prefix-free codes only
  config.max_scan_bits = tq::TieredQIVFIndex::automatic_scan_bits(config);
  const auto ids = codec_training_sample(size_t(n));
  config.num_centroids = std::min(maximum_centroids, uint32_t(ids.size() / 2));
  install(std::make_shared<const tq::TieredQFastBlocks>(
      tq::TieredQ::train_sampled(config, x, size_t(n), ids.data(), ids.size(), training_seed)));
}

void IndexIVFTieredQ::encode_vectors(idx_t n, const float* x, const idx_t* list_nos, uint8_t* codes,
                                     bool include_listno) const {
  if (!include_listno) { codec().encode(x, size_t(n), codes); return; }
  const size_t coarse = coarse_code_size();
  const auto rows = codec().encode(x, size_t(n));
  for (idx_t i = 0; i < n; ++i) {
    uint8_t* code = codes + i * (coarse + code_size);
    if (list_nos[i] < 0) { std::memset(code, 0, coarse + code_size); continue; }
    encode_listno(list_nos[i], code);
    std::memcpy(code + coarse, rows.data() + i * code_size, code_size);
  }
}

void IndexIVFTieredQ::decode_vectors(idx_t n, const uint8_t* codes, const idx_t*, float* x) const {
  codec().decode(codes, size_t(n), x);
}

void IndexIVFTieredQ::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
  const size_t coarse = coarse_code_size();
  std::vector<uint8_t> rows(size_t(n) * code_size);
  for (idx_t i = 0; i < n; ++i) std::memcpy(rows.data() + i * code_size, bytes + i * (coarse + code_size) + coarse, code_size);
  codec().decode(rows.data(), size_t(n), x);
}

void IndexIVFTieredQ::add_codes(idx_t n, const uint8_t* codes, const idx_t* list_nos, const idx_t* ids) {
  auto& lists = this->lists();
  // Rows grouped by list, input order kept within each (a counting sort).
  std::vector<idx_t> first(nlist + 1, 0);
  for (idx_t i = 0; i < n; ++i) if (list_nos[i] >= 0) ++first[list_nos[i] + 1];
  std::partial_sum(first.begin(), first.end(), first.begin());
  std::vector<idx_t> order(first[nlist]);
  {
    std::vector<idx_t> next(first.begin(), first.end() - 1);
    for (idx_t i = 0; i < n; ++i) if (list_nos[i] >= 0) order[next[list_nos[i]]++] = i;
  }
  std::vector<size_t> old_size(nlist);
  lists.admit_ids(ids, size_t(n));
  // Lists are independent, so each thread grows and fills its own.
#pragma omp parallel for schedule(dynamic, 16)
  for (idx_t list = 0; list < idx_t(nlist); ++list) {
    const idx_t begin = first[list], end = first[list + 1];
    old_size[list] = lists.list_size(list);
    if (begin == end) continue;
    const size_t new_size = old_size[list] + size_t(end - begin);
    if (new_size > lists.capacity(list)) lists.set_capacity(list, new_size);
    auto& stored = lists.lists[list];
    lists.reserve_ids(list, new_size);
    for (idx_t r = begin; r < end; ++r) {
      const size_t slot = lists.list_size(list);
      lists.append_ids(list, &ids[order[r]], 1);
      blocks_->write_row(stored.block.data(), stored.capacity, slot, codes + order[r] * code_size);
    }
  }
  // Hashtable maps key rows by id; array maps by position and reject given ids.
  faiss::DirectMapAdd map(direct_map, size_t(n), direct_map.type == faiss::DirectMap::Hashtable ? ids : nullptr);
  for (idx_t list = 0; list < idx_t(nlist); ++list)
    for (idx_t r = first[list]; r < first[list + 1]; ++r) map.add(size_t(order[r]), list, old_size[list] + size_t(r - first[list]));
}

void IndexIVFTieredQ::add_core(idx_t n, const float* x, const idx_t* xids, const idx_t* precomputed_idx, void*) {
  FAISS_THROW_IF_NOT(is_trained);
  direct_map.check_can_add(xids);
  std::vector<uint8_t> codes(size_t(n) * code_size);
  codec().encode(x, size_t(n), codes.data());
  std::vector<idx_t> ids(n);
  for (idx_t i = 0; i < n; ++i) ids[i] = xids ? xids[i] : ntotal + i;
  add_codes(n, codes.data(), precomputed_idx, ids.data());
  ntotal += n;
}

void IndexIVFTieredQ::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
  FAISS_THROW_IF_NOT(is_trained);
  std::vector<idx_t> list_nos(n);
  for (idx_t first = 0; first < n; first += assign_chunk_rows)
    quantizer->assign(std::min(assign_chunk_rows, n - first), x + first * d, list_nos.data() + first);
  add_core(n, x, xids, list_nos.data());
}

void IndexIVFTieredQ::merge_from(faiss::Index& other_index, idx_t add_id) {
  check_compatible_for_merge(other_index);
  auto& other = static_cast<IndexIVFTieredQ&>(other_index);
  FAISS_THROW_IF_NOT_MSG(other.codec().model_id() == codec().model_id() && other.blocks().format() == blocks().format(),
                         "can only merge IndexIVFTieredQ with the same codec and block format");
  auto& from = other.lists();
  for (size_t list = 0; list < nlist; ++list) {
    const size_t count = from.list_size(list);
    if (!count) continue;
    const auto rows = from.rows(list);
    std::vector<idx_t> list_nos(count, idx_t(list)), ids = from.ids(list);
    for (auto& id : ids) id += add_id;
    add_codes(idx_t(count), rows.data(), list_nos.data(), ids.data());
    from.resize(list, 0);
  }
  ntotal += other.ntotal;
  other.ntotal = 0;
}

size_t IndexIVFTieredQ::remove_ids(const faiss::IDSelector& sel) {
  auto& lists = this->lists();
  if (!direct_map.no()) {
    lists.use_wide_ids();
    return IndexIVF::remove_ids(sel);
  }
  size_t removed = 0;
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : removed)
  for (idx_t list = 0; list < idx_t(nlist); ++list) {
    auto ids = lists.ids(list);
    size_t live = ids.size();
    auto& stored = lists.lists[list];
    std::vector<uint8_t> row(code_size);
    for (size_t j = 0; j < live;) {
      if (!sel.is_member(ids[j])) { ++j; continue; }
      const uint32_t last = uint32_t(--live);
      ids[j] = ids[last];
      if (j == last) continue;
      lists.blocks->rows(stored.block.data(), stored.capacity, &last, 1, row.data());
      lists.blocks->write_row(stored.block.data(), stored.capacity, j, row.data());
    }
    if (live == ids.size()) continue;
    removed += ids.size() - live;
    lists.set_ids(list, 0, ids.data(), live);
    lists.resize(list, live);
  }
  ntotal -= idx_t(removed);
  return removed;
}

void IndexIVFTieredQ::reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons) const {
  faiss::InvertedLists::ScopedCodes row(invlists, size_t(list_no), size_t(offset));
  codec().decode(row.get(), 1, recons);
}

size_t IndexIVFTieredQ::storage_bytes() const { return codec().memory_bytes() + lists().allocated_bytes(); }

faiss::InvertedListScanner* IndexIVFTieredQ::get_InvertedListScanner(bool store_pairs, const faiss::IDSelector* sel,
                                                                    const faiss::IVFSearchParameters*) const {
  FAISS_THROW_IF_NOT_MSG(scanners_, "IndexIVFTieredQ is not trained");
  return new ListScanner(*this, scanners_, store_pairs, sel);
}
}  // namespace faiss_tieredq
