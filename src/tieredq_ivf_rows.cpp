// Exact payload rows of an IVF index's stored documents (#23). Fast-mode lists keep codes only in their scan layout:
// nibble lookup groups (narrow and additive x86 models), packed code words (wide tiers) or int8 levels (NEON). Each
// layout is inverted here, so rescoring tiers such as TieredQRefinement read their candidates' prefix rows from the
// lists instead of a second, row-major copy of the codes.
#include "tieredq/tieredq.h"
#include "tieredq_fast_storage.h"
#include <array>

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

// Split codes per width: packed part codes -> the whole code they were split from.
using SplitInverse=std::array<std::vector<uint8_t>,tieredq_detail::widest_code_bits+1>;
SplitInverse split_inverse(const tieredq_detail::AdditiveSplits& splits) {
  SplitInverse inverse;
  for (size_t bits=0;bits<splits.size();++bits) {
    if (!splits[bits]) continue;
    const auto& code=splits[bits]->code;
    inverse[bits].resize(code.size());
    for (size_t whole=0;whole<code.size();++whole) inverse[bits][code[whole]]=uint8_t(whole);
  }
  return inverse;
}

// Nibble groups hold whole codes or parts of split codes; parts of one coordinate are gathered before inverting.
void read_lookup_codes(const tieredq_detail::FastScanLookupCache& lookup,const SplitInverse& inverse,size_t doc,
                       std::vector<uint32_t>& packed,std::vector<const tieredq_detail::LookupSlot*>& split_slot,uint8_t* row) {
  using namespace tieredq_detail;
  std::fill(packed.begin(),packed.end(),0); std::fill(split_slot.begin(),split_slot.end(),nullptr);
  for (size_t g=0;g<lookup.groups.size();++g) {
    const auto& group=lookup.groups[g];
    const uint32_t nibble=uint32_t(lookup.codes[fast_lookup_position(lookup,doc,g)]>>(g%lookup_pair_groups*lookup_code_bits))&
        uint32_t(lookup_table_entries-1);
    const uint32_t mask=(1u<<group.bits)-1;
    for (uint32_t j=0;j<group.count;++j) {
      const auto& slot=group.slots[j];
      const uint32_t part=nibble>>(j*group.bits)&mask;
      if (slot.part==whole_code_part) { write_code(row,slot.bit,slot.code_bits,part); continue; }
      packed[slot.rank]|=part<<slot.part_shift;
      split_slot[slot.rank]=&slot;
    }
  }
  for (size_t rank=0;rank<split_slot.size();++rank) {
    const auto* slot=split_slot[rank];
    if (slot) write_code(row,slot->bit,slot->code_bits,inverse[slot->code_bits].at(packed[rank]));
  }
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
    const auto inverse=split_inverse(cache.lookup->splits);
    std::vector<uint32_t> packed(model.coded_dimensions());
    std::vector<const tieredq_detail::LookupSlot*> split_slot(model.coded_dimensions());
    for (size_t i=0;i<count;++i) read_lookup_codes(*cache.lookup,inverse,documents[i],packed,split_slot,rows+i*stride);
  } else if (cache.code_lookup()) {
    for (size_t i=0;i<count;++i) read_code_words(cache,documents[i],code_bytes,rows+i*stride);
  } else {
    const auto inverse=level_inverse(cache,model.levels_,model.tiers_);
    for (size_t i=0;i<count;++i) read_level_codes(cache,model.tiers_,inverse,documents[i],rows+i*stride);
  }
  for (size_t i=0;i<count;++i) write_metadata(cache,documents[i],code_bytes,rows+i*stride);
}
} // namespace dv
