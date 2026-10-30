// Fast-payload parsing shared by flat and IVF index loaders: validates the kind and level configuration against
// the model, then reads straight into the final cache layout.
#pragma once
#include "tieredq/tieredq.h"
#include "tieredq_fast_storage.h"

namespace dv {
template<class Reader> void TieredQIndex::read_fast_payload(Reader& reader,size_t rows) {
  using namespace tieredq_detail;
  if (!fast_mode_ || count_) bad("fast payload needs an empty fast index");
  if (reader.u32()!=fast_payload_version) bad("unsupported fast payload version");
  const uint32_t kind=reader.u32(),level_max=reader.u32();
  const bool byte_codes=kind==fast_byte_code_kind;
  const bool code_lookup=byte_codes || kind==fast_packed_code_kind;
  if (kind!=fast_lookup_kind && kind!=fast_levels_kind && !code_lookup) bad("invalid fast payload kind");
  const bool lookup=kind==fast_lookup_kind;
  if ((lookup && (level_max || !fast_lookup_supported(codec_.tiers_))) ||
      (code_lookup && level_max!=fast_code_lookup_levels) ||
      (!lookup && !code_lookup && level_max!=x86_fast_level_max && level_max!=arm_fast_level_max)) bad("invalid fast level configuration");
  auto storage=make_fast_storage(nullptr,0,lookup,int(level_max));
  read_fast_storage(reader,*storage,rows,codec_.config().num_centroids,byte_codes);
  fast_=std::move(storage);count_=rows;
}
} // namespace dv
