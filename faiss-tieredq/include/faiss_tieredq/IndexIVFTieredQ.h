// IndexIVFTieredQ (#55): a Faiss IVF index whose inverted lists hold TieredQ fast-scan blocks.
// - Encoder: a TieredQ codec (fast target, no fine tiers, the IVF scan cap) trained on a seeded sample of the training
//   rows, exactly as TieredQIVFIndex's Python build trains its codec; codes do not depend on the list
//   (by_residual = false), so merge_from, remove_ids and reconstruct stay exact. code_size is bytes_per_vector().
// - Lists: TieredQInvertedLists keeps each list in one TieredQFastBlocks block laid out for its capacity (its rows
//   rounded up to 16 documents after a bulk add, growing by half when full), so a list costs what TieredQIVFIndex's
//   packed lists cost. Fixed 256-document tiles would pad a 64K-list index of 8.84M rows to about twice its size.
// - Search: IndexIVF::search_preassigned with a scanner that scores each probed list in one call of the fast kernels.
//   Inner product returns TieredQ's score (larger is better); L2 returns squared distances (smaller is better);
//   cosine is inner product over rows and queries the codec normalizes (normalize the quantizer's inputs too, as usual
//   in Faiss).
// - Persistence: write_index_tieredq / read_index_tieredq (an out-of-tree type cannot hook into faiss::read_index);
//   the coarse quantizer is stored with faiss::write_index inside the file.
#pragma once
#include <faiss/IndexIVF.h>
#include <faiss/utils/AlignedTable.h>
#include <tieredq/tieredq.h>
#include <tieredq/tieredq_blocks.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace faiss_tieredq {
using faiss::idx_t;

// Faiss's InvertedLists contract over TieredQ blocks: codes go in and come out (get_single_code) as row-major encode()
// rows of code_size bytes, while get_codes() hands out the list's block, which only the TieredQ scanner reads.
// Ids are stored in 32 bits while every id lies in [0, 2^32), and widened to idx_t for get_ids(); the first id outside
// that range switches every list to 64-bit ids. Faiss's own lists keep 8 bytes per row, 6% of a 1-bit 1024-D row.
// A widened copy does not follow later updates, so IndexIVFTieredQ removes rows itself.
struct TieredQInvertedLists : faiss::InvertedLists {
  static constexpr size_t capacity_granularity = 16;  // documents per vector of the fast kernels
  TieredQInvertedLists(size_t nlist, std::shared_ptr<const tq::TieredQFastBlocks> blocks);

  size_t list_size(size_t list_no) const override;
  const uint8_t* get_codes(size_t list_no) const override { return lists[list_no].block.data(); }
  // 32-bit ids: a widened copy, freed by release_ids.
  const idx_t* get_ids(size_t list_no) const override;
  void release_ids(size_t list_no, const idx_t* ids) const override;
  idx_t get_single_id(size_t list_no, size_t offset) const override;
  // A decoded row, freed by release_codes.
  const uint8_t* get_single_code(size_t list_no, size_t offset) const override;
  void release_codes(size_t list_no, const uint8_t* codes) const override;
  size_t add_entries(size_t list_no, size_t n_entry, const idx_t* ids, const uint8_t* codes) override;
  void update_entries(size_t list_no, size_t offset, size_t n_entry, const idx_t* ids, const uint8_t* codes) override;
  void resize(size_t list_no, size_t new_size) override;

  // Lays the list out for rows_wanted (>= list_size) rounded up to capacity_granularity, keeping its rows.
  void set_capacity(size_t list_no, size_t rows_wanted);
  size_t capacity(size_t list_no) const { return lists[list_no].capacity; }
  // Switches to 64-bit ids if any of ids[0..n) does not fit 32 bits; call before writing those ids (not thread-safe).
  void admit_ids(const idx_t* ids, size_t n);
  // 64-bit ids from now on, so get_ids() points at the stored ids (not thread-safe).
  void use_wide_ids();
  bool wide_ids() const { return wide_ids_; }
  // The list's ids as idx_t and its rows, row-major.
  std::vector<idx_t> ids(size_t list_no) const;
  std::vector<uint8_t> rows(size_t list_no) const;
  // Appends n ids (already admitted) to the list's ids, or sets those at offset; the codes are written separately.
  void append_ids(size_t list_no, const idx_t* ids, size_t n);
  void set_ids(size_t list_no, size_t offset, const idx_t* ids, size_t n);
  void reserve_ids(size_t list_no, size_t n);
  size_t allocated_bytes() const;

  struct List {
    std::vector<uint32_t> narrow_ids;  // while !wide_ids()
    std::vector<idx_t> wide_ids;
    faiss::AlignedTableTightAlloc<uint8_t, tq::TieredQFastBlocks::alignment> block;  // block_bytes(capacity), zeroed
    size_t capacity = 0;
  };
  std::shared_ptr<const tq::TieredQFastBlocks> blocks;
  std::vector<List> lists;

 private:
  bool wide_ids_ = false;
};

// TieredQ scanners kept between searches: Faiss makes a list scanner per search and thread, and a fresh TieredQ scanner
// copies the format's geometry, so list scanners borrow one here and reset it per query.
class ScannerPool {
 public:
  explicit ScannerPool(std::shared_ptr<const tq::TieredQFastBlocks> blocks) : blocks_(std::move(blocks)) {}
  tq::TieredQFastBlocks::Scanner take(const float* query);
  void give(tq::TieredQFastBlocks::Scanner scanner);
  const tq::TieredQFastBlocks& blocks() const { return *blocks_; }

 private:
  std::shared_ptr<const tq::TieredQFastBlocks> blocks_;
  std::mutex mutex_;
  std::vector<tq::TieredQFastBlocks::Scanner> free_;
};

struct IndexIVFTieredQ : faiss::IndexIVF {
  // code_bits: the codec's bit budget per vector (e.g. d for 1 bit per dimension). metric: METRIC_INNER_PRODUCT or
  // METRIC_L2; cosine needs METRIC_INNER_PRODUCT.
  IndexIVFTieredQ(faiss::Index* quantizer, size_t d, size_t nlist, uint32_t code_bits,
                  faiss::MetricType metric = faiss::METRIC_L2, bool cosine = false);
  IndexIVFTieredQ();

  uint32_t code_bits = 0;
  bool cosine = false;

  // The trained codec; throws before training.
  const tq::TieredQ& codec() const;
  // Installs a codec trained elsewhere (e.g. the one a TieredQIVFIndex serves) instead of train_encoder's. format:
  // a persisted TieredQFastBlocks format, 0 for this CPU's. Only on an empty index.
  void set_codec(const tq::TieredQ& codec, uint32_t format = 0);
  const tq::TieredQFastBlocks& blocks() const;
  TieredQInvertedLists& lists() const;
  // Bytes of the lists' blocks and ids and the codec as allocated; excludes the quantizer.
  size_t storage_bytes() const;

  void train_encoder(idx_t n, const float* x, const idx_t* assign) override;
  void encode_vectors(idx_t n, const float* x, const idx_t* list_nos, uint8_t* codes,
                      bool include_listno = false) const override;
  void decode_vectors(idx_t n, const uint8_t* codes, const idx_t* list_nos, float* x) const override;
  void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
  void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;
  // precomputed_idx: the rows' lists (e.g. another index's assignment, so both hold the same partition).
  void add_core(idx_t n, const float* x, const idx_t* xids, const idx_t* precomputed_idx,
                void* inverted_list_context = nullptr) override;
  void merge_from(faiss::Index& other, idx_t add_id) override;
  // Each list moves its last row into a removed one's slot. With a direct map the lists switch to 64-bit ids and
  // Faiss's own removal runs.
  size_t remove_ids(const faiss::IDSelector& sel) override;
  void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons) const override;
  faiss::InvertedListScanner* get_InvertedListScanner(bool store_pairs = false, const faiss::IDSelector* sel = nullptr,
                                                      const faiss::IVFSearchParameters* params = nullptr) const override;

 private:
  void install(std::shared_ptr<const tq::TieredQFastBlocks> blocks);
  // Appends codes (code_size each) to their lists, in input order within each list; list_nos[i] < 0 skips row i.
  // Each list grows once, to exactly its new size.
  void add_codes(idx_t n, const uint8_t* codes, const idx_t* list_nos, const idx_t* ids);
  std::shared_ptr<const tq::TieredQFastBlocks> blocks_;
  std::shared_ptr<ScannerPool> scanners_;
};

// The rows (of count training rows) train_encoder trains the codec on: a seeded sample of at most 32,768, all of them
// in order when fewer. Training on just those rows, gathered, gives the same codec (e.g. from a store too large to
// decode whole).
std::vector<uint32_t> codec_training_sample(size_t count);

void write_index_tieredq(const IndexIVFTieredQ& index, const std::string& path);
// The caller owns the returned index (and through it its quantizer).
IndexIVFTieredQ* read_index_tieredq(const std::string& path);
}  // namespace faiss_tieredq
