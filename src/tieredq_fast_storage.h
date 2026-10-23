// Fast-only representation ownership, transactional append and versioned cache persistence.
#pragma once
#include "tieredq_io.h"
#include "tieredq_scan_lookup_x86.h"

namespace dv::tieredq_detail {
constexpr uint32_t fast_payload_version=1,fast_lookup_kind=1,fast_levels_kind=2;
constexpr size_t fast_metadata_bytes=sizeof(float)*2+sizeof(uint8_t);
inline size_t fast_storage_bytes(const FastScanCache& cache) {
  return cache.levels.capacity()+cache.factor.capacity()*sizeof(float)+cache.correction.capacity()*sizeof(float)+
      cache.centroid.capacity()+(cache.lookup ? cache.lookup->codes.capacity() : 0);
}
inline size_t fast_vector_bytes(const FastScanCache& cache) {
  return (cache.lookup ? cache.lookup->pairs() : cache.groups*4)+fast_metadata_bytes;
}
inline std::shared_ptr<FastScanCache> copy_fast_storage(const FastScanCache& cache) {
  auto copy=std::make_shared<FastScanCache>(cache);
  if (cache.lookup) copy->lookup=std::make_shared<FastScanLookupCache>(*cache.lookup);
  return copy;
}
inline void append_fast_rows(FastScanCache& cache,const std::vector<std::vector<float>>& books,
    const std::vector<TieredQTier>& tiers,size_t stride,size_t code_bytes,size_t centroids,
    const uint8_t* rows,size_t count) {
  if (!count) return;
  const size_t total=cache.count+count,block_docs=cache.lookup ? lookup_block_docs : fast_scan_block_docs;
  const size_t blocks=(total+block_docs-1)/block_docs,padded=blocks*block_docs;
  const Layout layout{code_bytes};
  std::vector<std::vector<int8_t>> levels(tiers.size());
  if (!cache.lookup) for (size_t ti=0;ti<tiers.size();++ti) {
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
  else reserve(cache.levels,product(product(blocks,cache.groups),64));
  cache.factor.resize(padded,0);cache.correction.resize(padded,0);cache.centroid.resize(padded,0);
  if (cache.lookup) cache.lookup->codes.resize(product(product(blocks,cache.lookup->pairs()),block_docs),0);
  else cache.levels.resize(product(product(blocks,cache.groups),64),0);
  for (size_t i=0;i<count;++i) {
    const uint8_t* row=rows+i*stride;const size_t doc=cache.count+i;
    cache.factor[doc]=layout.factor(row);cache.correction[doc]=layout.correction(row);cache.centroid[doc]=row[code_bytes];
    if (cache.lookup) {
      auto& lookup=*cache.lookup;
      for (size_t g=0;g<lookup.groups.size();++g) {
        const auto& group=lookup.groups[g];
        const uint8_t code=uint8_t(read_code(row,group.bit,group.count*group.bits));
        const size_t position=fast_lookup_position(lookup,doc,g);
        const size_t shift=g%lookup_pair_groups*lookup_code_bits;
        lookup.codes[position]=uint8_t((lookup.codes[position]&~((lookup_table_entries-1)<<shift))|(code<<shift));
      }
    } else for (size_t rank=0;rank<cache.dimensions;++rank) {
      const size_t ti=cache.tier_of[rank];const auto& tier=tiers[ti];
      const auto code=read_code(row,tier.bit_offset+(rank-tier.start)*tier.bits,tier.bits);
      cache.levels[fast_level_position(cache.groups,doc,rank)]=levels[ti][code];
    }
  }
  cache.count=total;cache.blocks=blocks;
}
template<class T> void write_fast_bytes(Writer& writer,const std::vector<T>& bytes) {
  static_assert(sizeof(T)==1);
  writer.u64(bytes.size());
  for (T value:bytes) writer.data.push_back(uint8_t(value));
}
template<class T> void read_fast_bytes(Reader& reader,std::vector<T>& bytes,size_t expected) {
  if (reader.u64()!=expected || expected>reader.data.size()-reader.pos) bad("invalid fast byte length");
  bytes.resize(expected);
  if (expected) std::memcpy(bytes.data(),reader.data.data()+reader.pos,expected);
  reader.pos+=expected;
}
inline std::vector<uint8_t> serialize_fast_storage(const FastScanCache& cache) {
  Writer writer;writer.u32(fast_payload_version);writer.u32(cache.lookup ? fast_lookup_kind : fast_levels_kind);
  writer.u32(cache.lookup ? 0 : uint32_t(cache.level_max));
  writer.floats(cache.factor);writer.floats(cache.correction);write_fast_bytes(writer,cache.centroid);
  if (cache.lookup) write_fast_bytes(writer,cache.lookup->codes); else write_fast_bytes(writer,cache.levels);
  return std::move(writer.data);
}
inline void read_fast_storage(Reader& reader,FastScanCache& cache,size_t count,size_t centroids) {
  const size_t block_docs=cache.lookup ? lookup_block_docs : fast_scan_block_docs;
  cache.count=count;cache.blocks=(count+block_docs-1)/block_docs;
  const size_t padded=product(cache.blocks,block_docs);
  auto floats=[&](std::vector<float>& values) {
    if (reader.u64()!=padded || padded>(reader.data.size()-reader.pos)/sizeof(float)) bad("invalid fast metadata length");
    values.resize(padded);for (auto& value:values) value=reader.f32();
  };
  floats(cache.factor);floats(cache.correction);read_fast_bytes(reader,cache.centroid,padded);
  if (std::any_of(cache.centroid.begin(),cache.centroid.end(),[&](uint8_t id){return id>=centroids;})) bad("invalid fast centroid id");
  if (cache.lookup) read_fast_bytes(reader,cache.lookup->codes,product(product(cache.blocks,cache.lookup->pairs()),block_docs));
  else {
    read_fast_bytes(reader,cache.levels,product(product(cache.blocks,cache.groups),64));
    if (std::any_of(cache.levels.begin(),cache.levels.end(),[&](int value){return value<-cache.level_max || value>cache.level_max;})) bad("invalid fast level");
  }
  if (reader.pos!=reader.data.size()) bad("trailing fast payload bytes");
}
} // namespace dv::tieredq_detail
