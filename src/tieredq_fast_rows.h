// Exact codec rows of documents held in fast storage (#23): fast layouts keep codes only in their scan layout - nibble
// lookup groups (narrow and additive x86 models), packed code words (wide tiers) or int8 levels (NEON) - and each is
// inverted here, for IVF list recovery and for TieredQFastBlocks::unpack.
#pragma once
#include "tieredq/tieredq.h"
#include "tieredq_fast_storage.h"
#include <array>
#include <numeric>

namespace dv::tieredq_detail {
namespace fast_rows {
constexpr int16_t no_code=-1,ambiguous_code=-2;
constexpr size_t int8_values=size_t(1)<<CHAR_BIT;

inline void write_metadata(const FastScanCache& cache,size_t doc,size_t code_bytes,uint8_t* row) {
  const Layout layout{code_bytes};
  auto store=[&](size_t offset,uint16_t bits) { row[offset]=uint8_t(bits); row[offset+1]=uint8_t(bits>>CHAR_BIT); };
  row[layout.centroid_offset()]=cache.centroid[doc];
  store(layout.factor_offset(),cache.factor[doc]);
  store(layout.correction_offset(),cache.correction[doc]);
}

inline void read_code_words(const FastScanCache& cache,size_t doc,size_t code_bytes,uint8_t* row) {
  const size_t words=cache.code_words(),stored=std::min(code_bytes,words*fast_code_word_bytes);
  for (size_t byte=0;byte<stored;++byte)
    row[byte]=std::bit_cast<uint8_t>(cache.levels[fast_code_word_position(words,doc,byte/fast_code_word_bytes)+byte%fast_code_word_bytes]);
}

// Nibble-lookup decoding flattened once per call: each group's slots as (rank, nibble shift, mask, part shift), then
// per rank the bit offset, width and the table mapping its gathered part codes to the stored code (the identity for
// whole codes, the inverse of the additive split otherwise).
struct LookupProgram {
  struct Slot { uint32_t rank, shift, mask, part_shift; };
  std::vector<Slot> slots;
  std::vector<uint32_t> group_end;  // exclusive end of each group's slots
  std::vector<uint32_t> bit, width;
  std::vector<const uint8_t*> code_of;
  std::array<std::vector<uint8_t>,widest_code_bits+1> split_inverse, identity;
  bool contiguous = false;  // ranks' codes fill the row from bit 0 in rank order
};

inline void build_lookup_program(const FastScanLookupCache& lookup,size_t ranks,LookupProgram& program) {
  program.slots.clear(); program.group_end.clear();
  program.bit.assign(ranks,0); program.width.assign(ranks,0); program.code_of.assign(ranks,nullptr);
  for (size_t bits=0;bits<=widest_code_bits;++bits) {
    program.identity[bits].resize(size_t(1)<<bits);
    std::iota(program.identity[bits].begin(),program.identity[bits].end(),uint8_t(0));
    program.split_inverse[bits].clear();
    if (bits>=lookup.splits.size() || !lookup.splits[bits]) continue;
    const auto& code=lookup.splits[bits]->code;
    program.split_inverse[bits].resize(code.size());
    for (size_t whole=0;whole<code.size();++whole) program.split_inverse[bits][code[whole]]=uint8_t(whole);
  }
  for (const auto& group:lookup.groups) {
    for (uint32_t j=0;j<group.count;++j) {
      const auto& slot=group.slots[j];
      const bool whole=slot.part==whole_code_part;
      program.slots.push_back({slot.rank,j*group.bits,(1u<<group.bits)-1,whole ? 0 : slot.part_shift});
      program.bit[slot.rank]=slot.bit; program.width[slot.rank]=slot.code_bits;
      program.code_of[slot.rank]=whole ? program.identity[slot.code_bits].data() : program.split_inverse[slot.code_bits].data();
    }
    program.group_end.push_back(uint32_t(program.slots.size()));
  }
  program.contiguous=true;
  for (size_t rank=0,next=0;rank<ranks;next+=program.width[rank],++rank) {
    if (!program.code_of[rank]) bad("lookup layout leaves a coded rank unstored");
    program.contiguous&=program.bit[rank]==next;
  }
}

// Byte stores alias everything, so the tables are read through local restrict pointers; contiguous codes stream
// through a 64-bit accumulator instead of read-modify-writes of shared bytes.
inline void read_lookup_codes(const FastScanLookupCache& lookup,const LookupProgram& program,size_t doc,
                       std::vector<uint32_t>& packed,uint8_t* row) {
  uint32_t* __restrict parts=packed.data();
  const size_t ranks=packed.size(),groups=program.group_end.size();
  std::fill_n(parts,ranks,0);
  const size_t stride=lookup.stride(doc/lookup_block_docs);
  const uint8_t* block=lookup.codes.data()+doc/lookup_block_docs*lookup.pairs()*lookup_block_docs+doc%lookup_block_docs;
  const LookupProgram::Slot* __restrict slots=program.slots.data();
  const uint32_t* __restrict group_end=program.group_end.data();
  for (size_t g=0,slot=0;g<groups;++g) {
    const uint32_t nibble=uint32_t(block[g/lookup_pair_groups*stride])>>(g%lookup_pair_groups*lookup_code_bits);
    for (const size_t end=group_end[g];slot<end;++slot) parts[slots[slot].rank]|=(nibble>>slots[slot].shift&slots[slot].mask)<<slots[slot].part_shift;
  }
  const uint32_t* __restrict bit=program.bit.data();
  const uint32_t* __restrict width=program.width.data();
  const uint8_t* const* __restrict code_of=program.code_of.data();
  if (!program.contiguous) {
    for (size_t rank=0;rank<ranks;++rank) write_code(row,bit[rank],width[rank],code_of[rank][parts[rank]]);
    return;
  }
  uint64_t pending=0; uint32_t filled=0;
  for (size_t rank=0;rank<ranks;++rank) {
    pending|=uint64_t(code_of[rank][parts[rank]])<<filled;
    filled+=width[rank];
    for (;filled>=CHAR_BIT;filled-=CHAR_BIT,pending>>=CHAR_BIT) *row++=uint8_t(pending);
  }
  if (filled) *row=uint8_t(pending);
}

// Int8 level -> code per tier, as the cache builders quantized each tier's codebook.
inline std::vector<std::array<int16_t,int8_values>> level_inverse(const FastScanCache& cache,
    const std::vector<std::vector<float>>& books,const std::vector<TieredQTier>& tiers) {
  std::vector<std::array<int16_t,int8_values>> inverse(tiers.size());
  for (size_t ti=0;ti<tiers.size();++ti) {
    inverse[ti].fill(no_code);
    const auto& book=books.at(tiers[ti].bits);
    for (size_t code=0;code<book.size();++code) {
      const long level=std::clamp<long>(std::lround(book[code]/cache.level_scale[ti]),-cache.level_max,cache.level_max);
      auto& slot=inverse[ti][uint8_t(int8_t(level))];
      slot=slot==no_code ? int16_t(code) : ambiguous_code;
    }
  }
  return inverse;
}

inline void read_level_codes(const FastScanCache& cache,const std::vector<TieredQTier>& tiers,
                      const std::vector<std::array<int16_t,int8_values>>& inverse,size_t doc,uint8_t* row) {
  for (size_t rank=0;rank<cache.dimensions;++rank) {
    const size_t ti=cache.tier_of[rank];
    const auto& tier=tiers[ti];
    const int16_t code=inverse[ti][uint8_t(cache.levels[fast_level_position(cache.groups,doc,rank)])];
    // Distinct codebook entries that quantize to one level cannot be told apart.
    if (code<0) bad("fast levels do not identify the stored codes");
    write_code(row,tier.bit_offset+uint32_t(rank-tier.start)*tier.bits,tier.bits,unsigned(code));
  }
}
} // namespace fast_rows

// Writes count rows (bytes_per_vector() each) of the documents docs[0..count) stored in cache, a model's fast storage.
inline void read_fast_rows(const FastScanCache& cache,const std::vector<std::vector<float>>& levels,
                           const std::vector<TieredQTier>& tiers,size_t ranks,size_t code_bytes,size_t stride,
                           const uint32_t* docs,size_t count,uint8_t* rows) {
  using namespace fast_rows;
  std::fill_n(rows,count*stride,uint8_t(0));
  if (cache.lookup) {
    thread_local LookupProgram program;
    thread_local std::vector<uint32_t> packed;
    build_lookup_program(*cache.lookup,ranks,program);
    packed.resize(ranks);
    for (size_t i=0;i<count;++i) read_lookup_codes(*cache.lookup,program,docs[i],packed,rows+i*stride);
  } else if (cache.code_lookup()) {
    for (size_t i=0;i<count;++i) read_code_words(cache,docs[i],code_bytes,rows+i*stride);
  } else {
    const auto inverse=level_inverse(cache,levels,tiers);
    for (size_t i=0;i<count;++i) read_level_codes(cache,tiers,inverse,docs[i],rows+i*stride);
  }
  for (size_t i=0;i<count;++i) write_metadata(cache,docs[i],code_bytes,rows+i*stride);
}
} // namespace dv::tieredq_detail
