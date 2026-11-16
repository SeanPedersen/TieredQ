// IndexIVFTieredQ files: a header, the coarse quantizer as faiss::write_index bytes, the TieredQ codec and block format,
// then every list's ids and block as stored. Native byte order (little endian on the supported platforms); the block
// format makes another CPU family scan the blocks with TieredQ's portable scorer.
//   magic u32 | version u32 | d u64 | nlist u64 | metric u32 | code_bits u32 | cosine u32 | ntotal u64
//   quantizer: bytes u64, data | codec: bytes u64, data | format u32
//   wide ids u32 | per list: count u64, capacity u64, ids (u32 or, when wide, i64)[count], block u8[block_bytes(capacity)]
#include "faiss_tieredq/IndexIVFTieredQ.h"
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <cstdio>
#include <memory>
#include <vector>

namespace faiss_tieredq {
namespace {
constexpr uint32_t file_magic = 0x56465154;  // "TQFV"
constexpr uint32_t file_version = 1;

struct File {
  File(const std::string& path, const char* mode) : handle(std::fopen(path.c_str(), mode)) {
    FAISS_THROW_IF_NOT_FMT(handle, "cannot open %s", path.c_str());
  }
  ~File() { std::fclose(handle); }
  void write(const void* data, size_t bytes) {
    FAISS_THROW_IF_NOT_MSG(std::fwrite(data, 1, bytes, handle) == bytes, "IndexIVFTieredQ write failed");
  }
  void read(void* data, size_t bytes) {
    FAISS_THROW_IF_NOT_MSG(std::fread(data, 1, bytes, handle) == bytes, "truncated IndexIVFTieredQ file");
  }
  template<class T> void put(T value) { write(&value, sizeof(value)); }
  template<class T> T get() { T value; read(&value, sizeof(value)); return value; }
  void put_bytes(const std::vector<uint8_t>& bytes) { put<uint64_t>(bytes.size()); write(bytes.data(), bytes.size()); }
  std::vector<uint8_t> get_bytes() {
    std::vector<uint8_t> bytes(get<uint64_t>());
    read(bytes.data(), bytes.size());
    return bytes;
  }
  std::FILE* handle;
};
}  // namespace

void write_index_tieredq(const IndexIVFTieredQ& index, const std::string& path) {
  const auto& lists = index.lists();
  File file(path, "wb");
  file.put(file_magic); file.put(file_version);
  file.put<uint64_t>(index.d); file.put<uint64_t>(index.nlist); file.put<uint32_t>(index.metric_type);
  file.put(index.code_bits); file.put<uint32_t>(index.cosine); file.put<uint64_t>(index.ntotal);
  faiss::VectorIOWriter quantizer;
  faiss::write_index(index.quantizer, &quantizer);
  file.put_bytes(quantizer.data);
  file.put_bytes(index.codec().serialize());
  file.put(index.blocks().format());
  file.put<uint32_t>(lists.wide_ids());
  for (size_t list = 0; list < index.nlist; ++list) {
    const auto& stored = lists.lists[list];
    file.put<uint64_t>(lists.list_size(list)); file.put<uint64_t>(stored.capacity);
    if (lists.wide_ids()) file.write(stored.wide_ids.data(), stored.wide_ids.size() * sizeof(idx_t));
    else file.write(stored.narrow_ids.data(), stored.narrow_ids.size() * sizeof(uint32_t));
    file.write(stored.block.data(), stored.block.nbytes());
  }
}

IndexIVFTieredQ* read_index_tieredq(const std::string& path) {
  File file(path, "rb");
  FAISS_THROW_IF_NOT_MSG(file.get<uint32_t>() == file_magic, "not an IndexIVFTieredQ file");
  FAISS_THROW_IF_NOT_MSG(file.get<uint32_t>() == file_version, "unsupported IndexIVFTieredQ file version");
  const auto d = file.get<uint64_t>(), nlist = file.get<uint64_t>();
  const auto metric = faiss::MetricType(file.get<uint32_t>());
  const auto code_bits = file.get<uint32_t>();
  const bool cosine = file.get<uint32_t>() != 0;
  const auto ntotal = idx_t(file.get<uint64_t>());
  faiss::VectorIOReader quantizer_bytes;
  quantizer_bytes.data = file.get_bytes();
  std::unique_ptr<faiss::Index> quantizer(faiss::read_index(&quantizer_bytes));
  FAISS_THROW_IF_NOT_MSG(quantizer->d == idx_t(d) && quantizer->ntotal == idx_t(nlist), "quantizer does not fit the index");
  const auto codec = tq::TieredQ::deserialize(file.get_bytes());
  const auto format = file.get<uint32_t>();
  auto index = std::make_unique<IndexIVFTieredQ>(quantizer.get(), d, nlist, code_bits, metric, cosine);
  index->own_fields = true;
  quantizer.release();
  index->set_codec(codec, format);
  auto& lists = index->lists();
  if (file.get<uint32_t>()) lists.use_wide_ids();
  for (size_t list = 0; list < nlist; ++list) {
    const auto count = file.get<uint64_t>(), capacity = file.get<uint64_t>();
    FAISS_THROW_IF_NOT_MSG(count <= capacity && capacity % TieredQInvertedLists::capacity_granularity == 0,
                           "invalid IndexIVFTieredQ list");
    lists.set_capacity(list, capacity);
    auto& stored = lists.lists[list];
    if (lists.wide_ids()) { stored.wide_ids.resize(count); file.read(stored.wide_ids.data(), count * sizeof(idx_t)); }
    else { stored.narrow_ids.resize(count); file.read(stored.narrow_ids.data(), count * sizeof(uint32_t)); }
    file.read(stored.block.data(), stored.block.nbytes());
  }
  index->ntotal = ntotal;
  return index.release();
}
}  // namespace faiss_tieredq
