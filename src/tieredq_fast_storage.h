// Fast-only representation ownership, transactional append and versioned cache persistence.
// Code-lookup caches persist packed code rows; legacy byte-per-coordinate payloads are repacked on load.
#pragma once
#include "tieredq_io.h"
#include "tieredq_scan_lookup_x86.h"
#include <climits>

namespace dv::tieredq_detail {
// Version 3 payloads (#26) hold fine-tier indexes: version 2's prefix storage, then every document's fine bits.
constexpr uint32_t fine_payload_version=3;
constexpr uint32_t fast_payload_version=2,fast_lookup_kind=1,fast_levels_kind=2,fast_byte_code_kind=3,fast_packed_code_kind=4;
constexpr size_t fast_metadata_bytes=sizeof(uint16_t)*2+sizeof(uint8_t);
inline size_t fast_storage_bytes(const FastScanCache& cache) {
  size_t bytes=sizeof(cache)+cache.levels.capacity()+cache.factor.capacity()*sizeof(uint16_t)+cache.correction.capacity()*sizeof(uint16_t)+
      cache.centroid.capacity()+(cache.lookup ? cache.lookup->allocated_bytes() : 0);
  bytes+=cache.code_offset.capacity()*sizeof(uint32_t)+cache.level_scale.capacity()*sizeof(float)+cache.tier_of.capacity();
  bytes+=cache.codebooks.capacity()*sizeof(std::vector<float>);
  for (const auto& book:cache.codebooks) bytes+=book.capacity()*sizeof(float);
  return bytes;
}
inline size_t fast_vector_bytes(const FastScanCache& cache) {
  return (cache.lookup ? cache.lookup->pairs() : cache.level_block_bytes()/fast_scan_block_docs)+fast_metadata_bytes;
}
inline std::shared_ptr<FastScanCache> copy_fast_storage(const FastScanCache& cache) {
  auto copy=std::make_shared<FastScanCache>(cache);
  if (cache.lookup) copy->lookup=std::make_shared<FastScanLookupCache>(*cache.lookup);
  return copy;
}
// Exact capacity for total rows, so appends up to total never reallocate.
inline void reserve_fast_rows(FastScanCache& cache,size_t total) {
  const size_t block_docs=cache.lookup ? lookup_block_docs : fast_scan_block_docs;
  const size_t blocks=(total+block_docs-1)/block_docs,padded=blocks*block_docs;
  cache.factor.reserve(padded);cache.correction.reserve(padded);cache.centroid.reserve(padded);
  if (cache.lookup) cache.lookup->codes.reserve(product(product(blocks,cache.lookup->pairs()),block_docs));
  else cache.levels.reserve(product(blocks,cache.level_block_bytes()));
}
inline void append_fast_rows(FastScanCache& cache,const std::vector<std::vector<float>>& books,
    const std::vector<TieredQTier>& tiers,size_t stride,size_t code_bytes,size_t centroids,
    const uint8_t* rows,size_t count) {
  if (!count) return;
  const size_t total=cache.count+count,block_docs=cache.lookup ? lookup_block_docs : fast_scan_block_docs;
  const size_t blocks=(total+block_docs-1)/block_docs,padded=blocks*block_docs;
  const Layout layout{code_bytes};
  std::vector<std::vector<int8_t>> levels(tiers.size());
  if (!cache.lookup && !cache.code_lookup()) for (size_t ti=0;ti<tiers.size();++ti) {
    const auto& book=books[tiers[ti].bits]; levels[ti].resize(book.size());
    for (size_t code=0;code<book.size();++code)
      levels[ti][code]=int8_t(std::clamp<long>(std::lround(book[code]/cache.level_scale[ti]),-cache.level_max,cache.level_max));
  }
  for (size_t i=0;i<count;++i) {
    if (rows[i*stride+code_bytes]>=centroids) bad("invalid fast centroid id");
    layout.validate_payload(rows+i*stride);
  }
  auto reserve=[](auto& values,size_t required) {
    if (required<=values.capacity()) return;
    constexpr size_t growth_divisor=2;
    const size_t headroom=values.capacity()/growth_divisor;
    const size_t grown=headroom>values.max_size()-values.capacity() ? required : values.capacity()+headroom;
    values.reserve(std::max(required,grown));
  };
  reserve(cache.factor,padded);reserve(cache.correction,padded);reserve(cache.centroid,padded);
  if (cache.lookup) reserve(cache.lookup->codes,product(product(blocks,cache.lookup->pairs()),block_docs));
  else reserve(cache.levels,product(blocks,cache.level_block_bytes()));
  cache.factor.resize(padded,0);cache.correction.resize(padded,0);cache.centroid.resize(padded,0);
  if (cache.lookup) cache.lookup->codes.resize(product(product(blocks,cache.lookup->pairs()),block_docs),0);
  else cache.levels.resize(product(blocks,cache.level_block_bytes()),0);
  for (size_t i=0;i<count;++i) {
    const uint8_t* row=rows+i*stride;const size_t doc=cache.count+i;
    cache.factor[doc]=layout.half(row,layout.factor_offset());cache.correction[doc]=layout.half(row,layout.correction_offset());cache.centroid[doc]=row[code_bytes];
    if (cache.lookup) {
      auto& lookup=*cache.lookup;
      for (size_t g=0;g<lookup.groups.size();++g) {
        const auto& group=lookup.groups[g];
        const uint8_t code=uint8_t(lookup_group_code(lookup,group,row));
        const size_t position=fast_lookup_position(lookup,doc,g);
        const size_t shift=g%lookup_pair_groups*lookup_code_bits;
        lookup.codes[position]=uint8_t((lookup.codes[position]&~((lookup_table_entries-1)<<shift))|(code<<shift));
      }
    } else if (cache.code_lookup()) store_code_row(cache,doc,row);
    else for (size_t rank=0;rank<cache.dimensions;++rank) {
      const size_t ti=cache.tier_of[rank];const auto& tier=tiers[ti];
      const auto code=read_code(row,tier.bit_offset+(rank-tier.start)*tier.bits,tier.bits);
      cache.levels[fast_level_position(cache.groups,doc,rank)]=levels[ti][code];
    }
  }
  cache.count=total;cache.blocks=blocks;
}
template<class C> void write_fast_bytes(Writer& writer,const C& bytes) {
  using T=typename C::value_type;
  static_assert(sizeof(T)==1);
  writer.u64(bytes.size());
  for (T value:bytes) writer.data.push_back(uint8_t(value));
}
template<class R,class C> void read_fast_bytes(R& reader,C& bytes,size_t expected) {
  static_assert(sizeof(typename C::value_type)==1);
  if (reader.u64()!=expected || expected>reader.remaining()) bad("invalid fast byte length");
  bytes.resize(expected);
  reader.bytes(bytes.data(),expected);
}
inline std::vector<uint8_t> serialize_fast_storage(const FastScanCache& cache,uint32_t version=fast_payload_version) {
  Writer writer;writer.u32(version);
  writer.u32(cache.lookup ? fast_lookup_kind : cache.code_lookup() ? fast_packed_code_kind : fast_levels_kind);
  writer.u32(cache.lookup ? 0 : uint32_t(cache.level_max));
  writer.halves(cache.factor.owned());writer.halves(cache.correction.owned());write_fast_bytes(writer,cache.centroid);
  if (cache.lookup) write_fast_bytes(writer,cache.lookup->codes); else write_fast_bytes(writer,cache.levels);
  return std::move(writer.data);
}
// Legacy payloads store one byte per coordinate in 4-coordinate groups; padding documents are dropped.
inline void repack_byte_codes(FastScanCache& cache,const std::vector<int8_t>& bytes) {
  std::vector<uint8_t> row((cache.code_bits+CHAR_BIT-1)/CHAR_BIT);
  for (size_t doc=0;doc<cache.blocks*fast_scan_block_docs;++doc) {
    std::fill(row.begin(),row.end(),0);
    for (size_t rank=0;rank<cache.dimensions;++rank) {
      const uint8_t code=uint8_t(bytes[fast_level_position(cache.groups,doc,rank)]);
      const size_t width=fast_code_width(cache,rank);
      if (code>>width) bad("invalid fast code");
      write_code(row.data(),cache.code_offset[rank],unsigned(width),code);
    }
    if (doc<cache.count) store_code_row(cache,doc,row.data());
  }
}
// Packed words hold no bits past code_bits, and padding documents are all zero.
inline void validate_packed_codes(const FastScanCache& cache) {
  for (size_t doc=0;doc<cache.blocks*fast_scan_block_docs;++doc) for (size_t word=0;word<cache.code_words();++word) {
    const size_t used=doc<cache.count ? std::min(fast_code_word_bits,cache.code_bits-word*fast_code_word_bits) : 0;
    const uint64_t allowed=(uint64_t(1)<<used)-1;
    if (load_code_word(cache.levels.data()+fast_code_word_position(cache.code_words(),doc,word))&~allowed) bad("nonzero fast code padding");
  }
}
// Parses directly into the cache; R is an in-memory Reader or a file StreamReader.
template<class R> void read_fast_storage(R& reader,FastScanCache& cache,size_t count,size_t centroids,bool byte_codes=false) {
  const size_t block_docs=cache.lookup ? lookup_block_docs : fast_scan_block_docs;
  cache.count=count;cache.blocks=(count+block_docs-1)/block_docs;
  const size_t padded=product(cache.blocks,block_docs);
  auto halves=[&](BlockArray<uint16_t>& values) {
    if (reader.u64()!=padded || padded>reader.remaining()/sizeof(uint16_t)) bad("invalid fast metadata length");
    std::vector<uint8_t> bytes(product(padded,sizeof(uint16_t)));reader.bytes(bytes.data(),bytes.size());
    values.resize(padded);
    for (size_t i=0;i<padded;++i) values[i]=uint16_t(bytes[2*i]|uint16_t(bytes[2*i+1])<<8);
    constexpr uint16_t exponent_mask=0x7c00;
    if (std::any_of(values.begin(),values.end(),[](uint16_t v){return (v&exponent_mask)==exponent_mask;})) bad("nonfinite fast metadata");
  };
  halves(cache.factor);halves(cache.correction);read_fast_bytes(reader,cache.centroid,padded);
  if (std::any_of(cache.centroid.begin(),cache.centroid.end(),[&](uint8_t id){return id>=centroids;})) bad("invalid fast centroid id");
  if (cache.lookup) read_fast_bytes(reader,cache.lookup->codes,product(product(cache.blocks,cache.lookup->pairs()),block_docs));
  else if (byte_codes) {
    std::vector<int8_t> bytes;
    read_fast_bytes(reader,bytes,product(product(cache.blocks,cache.groups),fast_scan_group_bytes));
    cache.levels.assign(product(cache.blocks,cache.level_block_bytes()),0);
    repack_byte_codes(cache,bytes);
  } else {
    read_fast_bytes(reader,cache.levels,product(cache.blocks,cache.level_block_bytes()));
    if (cache.code_lookup()) validate_packed_codes(cache);
    else if (std::any_of(cache.levels.begin(),cache.levels.end(),[&](int value){return value<-cache.level_max || value>cache.level_max;})) bad("invalid fast level");
  }
}
} // namespace dv::tieredq_detail
