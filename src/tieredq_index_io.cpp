// Exact/fast index persistence: legacy packed rows or a self-contained fast representation.
#include "tieredq/tieredq.h"
#include "tieredq_fast_storage.h"
namespace dv {
using tieredq_detail::Layout;
using tieredq_detail::bad;
using tieredq_detail::product;
using tieredq_detail::save_file;
using tieredq_detail::load_file;
void TieredQIndex::save(const std::string& path) const {
  if (fast_mode_) {
    auto storage=fast_ ? fast_ : make_fast_storage(nullptr,0);
    save_file(path,codec_.serialize(),tieredq_detail::serialize_fast_storage(*storage),size(),tieredq_detail::fast_index_file_kind);
    return;
  }
  const size_t stride=codec_.bytes_per_vector();
  std::vector<uint8_t> rows(product(size(),stride));
  for (size_t i=0;i<size();++i) for (size_t byte=0;byte<stride;++byte)
    rows[i*stride+byte]=payload_[(i/16)*16*stride+byte*16+i%16];
  save_file(path,codec_.serialize(),rows,size(),tieredq_detail::exact_index_file_kind);
}
TieredQIndex TieredQIndex::load(const std::string& path) {
  auto file=load_file(path,0);
  if (file.kind!=tieredq_detail::exact_index_file_kind && file.kind!=tieredq_detail::fast_index_file_kind) bad("wrong index file kind");
  if (file.rows>UINT32_MAX) bad("index exceeds uint32 document ids");
  TieredQIndex index(TieredQ::deserialize(file.model),file.kind==tieredq_detail::fast_index_file_kind ? TieredQIndexMode::fast : TieredQIndexMode::full);
  if (index.fast_mode_) {
    tieredq_detail::Reader reader{file.payload};
    if (reader.u32()!=tieredq_detail::fast_payload_version) bad("unsupported fast payload version");
    const uint32_t kind=reader.u32(),level_max=reader.u32();
    const bool byte_codes=kind==tieredq_detail::fast_byte_code_kind;
    const bool code_lookup=byte_codes || kind==tieredq_detail::fast_packed_code_kind;
    if (kind!=tieredq_detail::fast_lookup_kind && kind!=tieredq_detail::fast_levels_kind && !code_lookup) bad("invalid fast payload kind");
    const bool lookup=kind==tieredq_detail::fast_lookup_kind;
    if ((lookup && (level_max || !tieredq_detail::fast_lookup_supported(index.codec_.tiers_))) ||
        (code_lookup && level_max!=tieredq_detail::fast_code_lookup_levels) ||
        (!lookup && !code_lookup && level_max!=tieredq_detail::x86_fast_level_max && level_max!=tieredq_detail::arm_fast_level_max)) bad("invalid fast level configuration");
    auto storage=index.make_fast_storage(nullptr,0,lookup,int(level_max));
    tieredq_detail::read_fast_storage(reader,*storage,size_t(file.rows),index.codec_.config().num_centroids,byte_codes);
    index.fast_=std::move(storage);index.count_=size_t(file.rows);
    return index;
  }
  if (file.rows>UINT32_MAX || product(size_t(file.rows),index.codec_.bytes_per_vector())!=file.payload.size()) bad("payload length mismatch");
  for (size_t i=0;i<file.rows;++i) {
    const auto* doc=file.payload.data()+i*index.codec_.bytes_per_vector();
    const Layout layout{index.codec_.code_bytes()};
    layout.validate_payload(doc);
    if (doc[layout.centroid_offset()]>=index.codec_.config().num_centroids) bad("invalid centroid id");
    if (index.codec_.config().code_bits%8 && (doc[index.codec_.code_bytes()-1]>>(index.codec_.config().code_bits%8))) bad("nonzero code padding");
  }
  const size_t stride=index.codec_.bytes_per_vector();
  index.count_=size_t(file.rows);
  index.payload_.resize(product(index.size()/16+(index.size()%16!=0),product(16,stride)),0);
  for (size_t i=0;i<index.size();++i) for (size_t byte=0;byte<stride;++byte)
    index.payload_[(i/16)*16*stride+byte*16+i%16]=file.payload[i*stride+byte];
  return index;
}
} // namespace dv
