// Fast-scan blocks in caller-owned memory: the fast representation of a TieredQIndex for one run of documents,
// serialized into a contiguous byte block that an engine stores (e.g. a DiskoVec extent on disk) and scans in place
// with the fast kernels, without copying or converting it per scan.
// - format() names the representation (nibble lookup, packed code words or int8 levels) chosen for the model and this
//   CPU. Persist it and reopen stored blocks with TieredQFastBlocks(codec, format): a CPU without that format's native
//   kernel scans it with the portable scorer, as cross-platform fast index loads do.
// - Blocks start at any address; starts on `alignment` bytes keep vector loads within cache lines.
// - Scores are model-space inner-product estimates, as TieredQ::score returns them; TieredQ::map_scores maps them to
//   the metric's scores.
// - Partly filled blocks (#55) serve engines that place one row at a time: a block laid out for `capacity` documents
//   (block_bytes(capacity)) is filled slot by slot with write_row and scanned over its first `live` slots, so a list
//   grows into spare capacity and an engine with fixed-size blocks uses one capacity for all (tile_slots).
#pragma once
#include "tieredq/tieredq.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tq {
namespace tieredq_detail { struct FastScanCache; class FastRowReader; class FastTileWriter; }
class TieredQFastBlocks {
 public:
  static constexpr size_t alignment = 64;
  // The format a fast TieredQIndex of this model would use on this CPU.
  explicit TieredQFastBlocks(const TieredQ& codec);
  // A persisted format; rejected when it does not belong to this model.
  TieredQFastBlocks(const TieredQ& codec, uint32_t format);
  const TieredQ& codec() const { return codec_; }
  uint32_t format() const { return format_; }
  // Bytes of a block holding count documents (count may be 0).
  size_t block_bytes(size_t count) const;
  // rows: count codec().encode() rows (bytes_per_vector() each); writes block_bytes(count) bytes.
  void pack(const uint8_t* rows, size_t count, uint8_t* block) const;
  // Exact inverse of pack: count encode() rows. Throws when int8 levels cannot identify a code (never for formats
  // this model was packed with by pack, which checks it).
  void unpack(const uint8_t* block, size_t count, uint8_t* rows) const;
  // unpack of selected documents only: rows of documents[0..n) (each < count) of a block packed with count documents,
  // e.g. a scan's candidates whose codes a refinement tier rescores.
  void rows(const uint8_t* block, size_t count, const uint32_t* documents, size_t n, uint8_t* rows) const;

  // Zero a block of block_bytes(capacity) bytes before its first write_row; slots never written, or vacated by the
  // caller, are padding that scans of the first `live` slots skip. rows() and unpack() with count = capacity read
  // written slots back exactly. A fixed-size tile is a block of tile_slots documents (one nibble-lookup block, no short
  // tail); block_bytes(tile_slots) is a multiple of alignment, so tiles laid end to end keep their alignment.
  static constexpr size_t tile_slots = 256;
  // Stores one encode() row in slot (< capacity) of a block laid out for capacity documents, replacing what the slot
  // held: the block's bytes then equal pack() of its capacity rows wherever pack() writes them.
  void write_row(uint8_t* block, size_t capacity, size_t slot, const uint8_t* row) const;

  // One query's prepared tables plus a scan view; not thread-safe, reusable across blocks of this format.
  class Scanner {
   public:
    ~Scanner();
    Scanner(Scanner&&) noexcept;
    Scanner& operator=(Scanner&&) noexcept;
    // Writes count scores for the block's documents (count as packed). The block must outlive the call only.
    void score(const uint8_t* block, size_t count, float* scores);
    // Writes live scores for slots [0, live) of a block laid out for capacity documents (live <= capacity), equal bit
    // for bit to score() of a block packed with those live rows.
    void score(const uint8_t* block, size_t capacity, size_t live, float* scores);
    struct Impl;
   private:
    friend class TieredQFastBlocks;
    explicit Scanner(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
  };
  // query: config().dimensions floats in input space (lifted to model space as TieredQ::query does).
  Scanner scanner(const float* query) const;
  // From the query's codec().query_weights() prepared by the caller, who reuses them (e.g. TieredQRefinement::query).
  Scanner scanner(const TieredQQuery& weights) const;
  // Prepares a scanner of these blocks for another query in place, keeping its scan view: a new scanner copies the
  // format's geometry (about 25 us at 1024 dimensions), so engines that scan many queries reuse scanners.
  void reset(Scanner& scanner, const float* query) const;
  void reset(Scanner& scanner, const TieredQQuery& weights) const;

 private:
  Scanner make_scanner(const float* model_query, const TieredQQuery* weights) const;
  void check_weights(const TieredQQuery& weights) const;
  // query in model space: itself for inner product, otherwise lifted into `lifted`.
  const float* model_query(const float* query, std::vector<float>& lifted) const;
  static std::shared_ptr<const tieredq_detail::FastRowReader> make_reader(const TieredQ& codec,
                                                                          const tieredq_detail::FastScanCache& geometry);
  static std::shared_ptr<const tieredq_detail::FastTileWriter> make_writer(
      const TieredQ& codec, const std::shared_ptr<const tieredq_detail::FastScanCache>& geometry);
  TieredQ codec_;
  uint32_t format_ = 0;
  std::shared_ptr<const tieredq_detail::FastScanCache> empty_;  // the format's geometry with no documents
  std::shared_ptr<const tieredq_detail::FastRowReader> reader_;  // unpack / rows decoding tables of that geometry
  std::shared_ptr<const tieredq_detail::FastTileWriter> writer_;  // write_row encoding tables of that geometry
};
} // namespace tq
