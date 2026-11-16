// One document's fast fields written in place into a caller's fixed-capacity tile (#55): the counterpart of
// FastRowReader for TieredQFastBlocks::write_row. A tile has the block layout of tile_docs documents (whole 256-document
// lookup blocks, 16-document level and code-word blocks), so each field's position follows from the same position
// helpers the cache builders use, and the stored bytes equal what pack() writes for the row at that document.
#pragma once
#include "tieredq/tieredq.h"
#include "tieredq_io.h"
#include "tieredq_layout.h"
#include "tieredq_scan_int8.h"
#include "tieredq_scan_lookup_x86.h"
#include <bit>
#include <cstring>
#include <memory>

namespace tq::tieredq_detail {
class FastTileWriter {
 public:
  // geometry_storage: the format's empty fast storage; code_offset: byte offset of the codes in a tile of tile_docs documents.
  FastTileWriter(std::shared_ptr<const FastScanCache> geometry_storage,const std::vector<std::vector<float>>& books,
                 const std::vector<TieredQTier>& tiers,size_t code_bytes,size_t centroids,size_t tile_docs,size_t code_offset)
      : storage_(std::move(geometry_storage)),geometry_(*storage_),tiers_(tiers),code_bytes_(code_bytes),centroids_(centroids),tile_docs_(tile_docs),code_offset_(code_offset) {
    if (geometry_.lookup || geometry_.code_lookup()) return;
    // The quantization build_fast_scan_cache applies to each tier's codebook.
    levels_.resize(tiers.size());
    for (size_t ti=0;ti<tiers.size();++ti) {
      const auto& book=books.at(tiers[ti].bits);
      levels_[ti].resize(book.size());
      for (size_t code=0;code<book.size();++code)
        levels_[ti][code]=int8_t(std::clamp<long>(std::lround(book[code]/geometry_.level_scale[ti]),-geometry_.level_max,geometry_.level_max));
    }
  }
  FastTileWriter(const FastTileWriter&)=delete;
  FastTileWriter& operator=(const FastTileWriter&)=delete;

  void write(uint8_t* tile,size_t slot,const uint8_t* row) const {
    if (row[code_bytes_]>=centroids_) bad("invalid fast centroid id");
    const Layout layout{code_bytes_};
    store_half(tile,slot,layout.half(row,layout.factor_offset()));
    store_half(tile+tile_docs_*sizeof(uint16_t),slot,layout.half(row,layout.correction_offset()));
    tile[tile_docs_*2*sizeof(uint16_t)+slot]=row[code_bytes_];
    uint8_t* codes=tile+code_offset_;
    if (geometry_.lookup) write_lookup(codes,slot,row);
    else if (geometry_.code_lookup()) write_code_words(codes,slot,row);
    else write_levels(codes,slot,row);
  }

 private:
  // Host-order binary16 bits, as pack() copies the cache's arrays and scans borrow them.
  static void store_half(uint8_t* values,size_t slot,uint16_t bits) { std::memcpy(values+slot*sizeof(uint16_t),&bits,sizeof(bits)); }

  // Each pair byte holds two groups' nibbles; the slot's other nibble belongs to a neighbouring group.
  void write_lookup(uint8_t* codes,size_t slot,const uint8_t* row) const {
    const auto& lookup=*geometry_.lookup;
    for (size_t g=0;g<lookup.groups.size();++g) {
      const uint8_t code=uint8_t(lookup_group_code(lookup,lookup.groups[g],row));
      const size_t shift=g%lookup_pair_groups*lookup_code_bits;
      uint8_t& pair=codes[fast_lookup_position(lookup,slot,g)];
      pair=uint8_t((pair&~((lookup_table_entries-1)<<shift))|(code<<shift));
    }
  }
  // store_code_row's bytes: the packed row with bits past code_bits cleared.
  void write_code_words(uint8_t* codes,size_t slot,const uint8_t* row) const {
    const size_t words=geometry_.code_words(),code_bits=geometry_.code_bits;
    for (size_t byte=0;byte<words*fast_code_word_bytes;++byte) {
      const size_t bit=byte*CHAR_BIT;
      const uint8_t value=bit<code_bits ? uint8_t(row[byte]&((1u<<std::min<size_t>(CHAR_BIT,code_bits-bit))-1)) : 0;
      codes[fast_code_word_position(words,slot,byte/fast_code_word_bytes)+byte%fast_code_word_bytes]=value;
    }
  }
  void write_levels(uint8_t* codes,size_t slot,const uint8_t* row) const {
    for (size_t rank=0;rank<geometry_.dimensions;++rank) {
      const size_t ti=geometry_.tier_of[rank];
      const auto& tier=tiers_[ti];
      const uint32_t code=read_code(row,tier.bit_offset+uint32_t(rank-tier.start)*tier.bits,tier.bits);
      codes[fast_level_position(geometry_.groups,slot,rank)]=std::bit_cast<uint8_t>(levels_[ti][code]);
    }
  }

  std::shared_ptr<const FastScanCache> storage_;
  const FastScanCache& geometry_;
  std::vector<TieredQTier> tiers_;
  size_t code_bytes_,centroids_,tile_docs_,code_offset_;
  std::vector<std::vector<int8_t>> levels_;  // level geometries: int8 level of each tier's codes
};
} // namespace tq::tieredq_detail
