// Exact/fast index persistence: legacy packed rows or a self-contained fast representation.
#include "tieredq/tieredq.h"
#include "tieredq_index_read.h"
#include <span>
namespace dv {
using tieredq_detail::Layout;
using tieredq_detail::bad;
using tieredq_detail::product;
using tieredq_detail::save_file;
std::vector<uint8_t> TieredQIndex::fast_payload() const {
  auto storage=fast_ ? fast_ : make_fast_storage(nullptr,0);
  auto payload=tieredq_detail::serialize_fast_storage(*storage,
      fine_tiers_ ? tieredq_detail::fine_payload_version : tieredq_detail::fast_payload_version);
  if (!fine_tiers_) return payload;
  tieredq_detail::Writer fine{std::move(payload)};
  tieredq_detail::write_fast_bytes(fine,std::span(fine_).first(product(size(),fine_tiers_->fine_bytes)));
  return std::move(fine.data);
}
void TieredQIndex::save(const std::string& path) const {
  if (fast_mode_) {
    save_file(path,codec_.serialize(),fast_payload(),size(),tieredq_detail::fast_index_file_kind);
    return;
  }
  const size_t stride=codec_.bytes_per_vector();
  std::vector<uint8_t> rows(product(size(),stride));
  for (size_t i=0;i<size();++i) for (size_t byte=0;byte<stride;++byte)
    rows[i*stride+byte]=payload_[(i/16)*16*stride+byte*16+i%16];
  save_file(path,codec_.serialize(),rows,size(),tieredq_detail::exact_index_file_kind);
}
// Parses the payload straight from the file into its final layout, so peak memory stays near
// the loaded index size. Payload corruption reports as a checksum failure, as with buffered loads.
TieredQIndex TieredQIndex::load(const std::string& path) {
  auto file=tieredq_detail::open_file(path,0);
  const auto& header=file.header;
  if (header.kind!=tieredq_detail::exact_index_file_kind && header.kind!=tieredq_detail::fast_index_file_kind) bad("wrong index file kind");
  if (header.rows>UINT32_MAX) bad("index exceeds uint32 document ids");
  tieredq_detail::StreamReader reader(file.in,size_t(header.payload_length));
  auto parse=[&] {
    TieredQIndex index(TieredQ::deserialize(file.model),header.kind==tieredq_detail::fast_index_file_kind ? TieredQIndexMode::fast : TieredQIndexMode::full);
    file.model=std::vector<uint8_t>();
    if (index.fast_mode_) {
      index.read_fast_payload(reader,size_t(header.rows));
      return index;
    }
    const size_t stride=index.codec_.bytes_per_vector();
    if (product(size_t(header.rows),stride)!=header.payload_length) bad("payload length mismatch");
    const Layout layout{index.codec_.code_bytes()};
    index.count_=size_t(header.rows);
    index.payload_.resize(product(index.size()/16+(index.size()%16!=0),product(16,stride)),0);
    std::vector<uint8_t> doc(stride);
    for (size_t i=0;i<index.size();++i) {
      reader.bytes(doc.data(),stride);
      layout.validate_payload(doc.data());
      if (doc[layout.centroid_offset()]>=index.codec_.config().num_centroids) bad("invalid centroid id");
      if (index.codec_.config().code_bits%8 && (doc[index.codec_.code_bytes()-1]>>(index.codec_.config().code_bits%8))) bad("nonzero code padding");
      for (size_t byte=0;byte<stride;++byte) index.payload_[(i/16)*16*stride+byte*16+i%16]=doc[byte];
    }
    return index;
  };
  try {
    auto index=parse();
    if (!reader.verify(header.payload_hash)) bad("checksum mismatch");
    return index;
  } catch (...) {
    if (!reader.verify(header.payload_hash)) bad("checksum mismatch");
    throw;
  }
}
} // namespace dv
