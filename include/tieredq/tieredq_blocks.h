// Fast-scan blocks in caller-owned memory: the fast representation of a TieredQIndex for one run of documents,
// serialized into a contiguous byte block that an engine stores (e.g. a DiskoVec extent on disk) and scans in place
// with the fast kernels, without copying or converting it per scan.
// - format() names the representation (nibble lookup, packed code words or int8 levels) chosen for the model and this
//   CPU. Persist it and reopen stored blocks with TieredQFastBlocks(codec, format): a CPU without that format's native
//   kernel scans it with the portable scorer, as cross-platform fast index loads do.
// - Blocks start at any address; starts on `alignment` bytes keep vector loads within cache lines.
// - Scores are model-space inner-product estimates, as TieredQ::score returns them; TieredQ::map_scores maps them to
//   the metric's scores.
// - Fixed-capacity tiles (#55) serve engines that place one row at a time into fixed-size blocks (a Faiss CodePacker):
//   a tile is a block of tile_slots documents, filled slot by slot with write_row and scanned with Scanner::score_tile.
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

  // A tile is block_bytes(tile_slots) bytes, a multiple of alignment, so tiles laid end to end keep their alignment.
  // Each carries its own metadata. Zero a tile before its first write_row; slots never written, or vacated by the
  // caller, are padding that scans skip. rows() and unpack() with count = tile_slots read written slots back exactly.
  static constexpr size_t tile_slots = 256;
  size_t tile_bytes() const;
  // Stores one encode() row in slot (< tile_slots) of tile, replacing what the slot held: the tile's bytes then equal
  // pack() of the tile's rows wherever pack() writes them.
  void write_row(uint8_t* tile, size_t slot, const uint8_t* row) const;

  // One query's prepared tables plus a scan view; not thread-safe, reusable across blocks of this format.
  class Scanner {
   public:
    ~Scanner();
    Scanner(Scanner&&) noexcept;
    Scanner& operator=(Scanner&&) noexcept;
    // Writes count scores for the block's documents (count as packed). The block must outlive the call only.
    void score(const uint8_t* block, size_t count, float* scores);
    // Writes live scores for slots [0, live) of a tile (live <= tile_slots), equal bit for bit to score() of a block
    // packed with those rows.
    void score_tile(const uint8_t* tile, size_t live, float* scores);
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

 private:
  Scanner make_scanner(const float* model_query, const TieredQQuery* weights) const;
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
