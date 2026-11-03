// Exact payload rows of an IVF index's stored documents (#23). Fast-mode lists keep codes only in their scan layout:
// nibble lookup groups (narrow and additive x86 models), packed code words (wide tiers) or int8 levels (NEON). Each
// layout is inverted here, so rescoring tiers such as TieredQRefinement read their candidates' prefix rows from the
// lists instead of a second, row-major copy of the codes.
#include "tieredq/tieredq.h"
#include "tieredq_fast_storage.h"
#include <array>
#include <numeric>

namespace dv {
using tieredq_detail::bad;
namespace {
constexpr uint32_t padding_id=UINT32_MAX;
constexpr int16_t no_code=-1,ambiguous_code=-2;
constexpr size_t int8_values=size_t(1)<<CHAR_BIT;

void write_metadata(const tieredq_detail::FastScanCache& cache,size_t doc,size_t code_bytes,uint8_t* row) {
  const tieredq_detail::Layout layout{code_bytes};
  auto store=[&](size_t offset,uint16_t bits) { row[offset]=uint8_t(bits); row[offset+1]=uint8_t(bits>>CHAR_BIT); };
  row[layout.centroid_offset()]=cache.centroid[doc];
  store(layout.factor_offset(),cache.factor[doc]);
  store(layout.correction_offset(),cache.correction[doc]);
}

void read_code_words(const tieredq_detail::FastScanCache& cache,size_t doc,size_t code_bytes,uint8_t* row) {
  using namespace tieredq_detail;
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
  std::array<std::vector<uint8_t>,tieredq_detail::widest_code_bits+1> split_inverse, identity;
  bool contiguous = false;  // ranks' codes fill the row from bit 0 in rank order
};

void build_lookup_program(const tieredq_detail::FastScanLookupCache& lookup,size_t ranks,LookupProgram& program) {
  using namespace tieredq_detail;
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
void read_lookup_codes(const tieredq_detail::FastScanLookupCache& lookup,const LookupProgram& program,size_t doc,
                       std::vector<uint32_t>& packed,uint8_t* row) {
  using namespace tieredq_detail;
  uint32_t* __restrict parts=packed.data();
  const size_t ranks=packed.size(),groups=program.group_end.size();
  std::fill_n(parts,ranks,0);
  const uint8_t* block=lookup.codes.data()+doc/lookup_block_docs*lookup.pairs()*lookup_block_docs+doc%lookup_block_docs;
  const LookupProgram::Slot* __restrict slots=program.slots.data();
  const uint32_t* __restrict group_end=program.group_end.data();
  for (size_t g=0,slot=0;g<groups;++g) {
    const uint32_t nibble=uint32_t(block[g/lookup_pair_groups*lookup_block_docs])>>(g%lookup_pair_groups*lookup_code_bits);
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
std::vector<std::array<int16_t,int8_values>> level_inverse(const tieredq_detail::FastScanCache& cache,
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

void read_level_codes(const tieredq_detail::FastScanCache& cache,const std::vector<TieredQTier>& tiers,
                      const std::vector<std::array<int16_t,int8_values>>& inverse,size_t doc,uint8_t* row) {
  using namespace tieredq_detail;
  for (size_t rank=0;rank<cache.dimensions;++rank) {
    const size_t ti=cache.tier_of[rank];
    const auto& tier=tiers[ti];
    const int16_t code=inverse[ti][uint8_t(cache.levels[fast_level_position(cache.groups,doc,rank)])];
    // Distinct codebook entries that quantize to one level cannot be told apart.
    if (code<0) bad("fast levels do not identify the stored codes");
    write_code(row,tier.bit_offset+uint32_t(rank-tier.start)*tier.bits,tier.bits,unsigned(code));
  }
}
} // namespace

std::vector<uint32_t> TieredQIVFIndex::id_documents() const {
  std::vector<uint32_t> documents(size_,padding_id);
  for (size_t doc=0;doc<ids_.size();++doc) if (ids_[doc]!=padding_id) documents[ids_[doc]]=uint32_t(doc);
  return documents;
}

void TieredQIVFIndex::document_rows(const uint32_t* documents,size_t count,uint8_t* rows) const {
  if (!count) return;
  if (!documents || !rows) bad("null document buffer");
  const TieredQ& model=codec();
  const size_t stride=model.bytes_per_vector(),code_bytes=model.code_bytes();
  for (size_t i=0;i<count;++i)
    if (documents[i]>=ids_.size() || ids_[documents[i]]==padding_id) bad("invalid stored document");
  std::fill_n(rows,count*stride,uint8_t(0));
  const auto& cache=*documents_.fast_;
  if (cache.lookup) {
    thread_local LookupProgram program;
    thread_local std::vector<uint32_t> packed;
    build_lookup_program(*cache.lookup,model.coded_dimensions(),program);
    packed.resize(model.coded_dimensions());
    for (size_t i=0;i<count;++i) read_lookup_codes(*cache.lookup,program,documents[i],packed,rows+i*stride);
  } else if (cache.code_lookup()) {
    for (size_t i=0;i<count;++i) read_code_words(cache,documents[i],code_bytes,rows+i*stride);
  } else {
    const auto inverse=level_inverse(cache,model.levels_,model.tiers_);
    for (size_t i=0;i<count;++i) read_level_codes(cache,model.tiers_,inverse,documents[i],rows+i*stride);
  }
  for (size_t i=0;i<count;++i) write_metadata(cache,documents[i],code_bytes,rows+i*stride);
}
} // namespace dv
