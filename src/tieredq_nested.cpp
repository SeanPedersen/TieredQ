// Nested IVF serving (#23): prefix scan, then prefix plus suffix rescoring of the k' candidates, plus the RAM and
// file row sources the tier is read from.
#include "tieredq/tieredq_nested.h"
#include "tieredq_io.h"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <numeric>
#include <sys/stat.h>
#include <unistd.h>

namespace dv {
using tieredq_detail::bad;
using tieredq_detail::product;
namespace {
constexpr uint32_t padding_id=UINT32_MAX;
constexpr uint32_t row_file_magic=0x57525154;  // "TQRW"
constexpr uint32_t nested_payload_version=2;
constexpr size_t encode_chunk_rows=16384;
constexpr size_t row_file_header_bytes=2*sizeof(uint32_t)+2*sizeof(uint64_t);

void check_row_id(uint32_t id,size_t rows) { if (id>=rows) bad("row id out of range"); }

std::string system_error(const char* what,const std::string& path) {
  return std::string(what)+" "+path+": "+std::strerror(errno);
}
void read_exact(int fd,uint8_t* out,size_t bytes,off_t offset) {
  while (bytes) {
    const ssize_t got=::pread(fd,out,bytes,offset);
    if (got<0 && errno==EINTR) continue;
    if (got<=0) bad(got<0 ? std::string("row read failed: ")+std::strerror(errno) : "truncated row file");
    out+=got; bytes-=size_t(got); offset+=got;
  }
}
void write_exact(int fd,const uint8_t* data,size_t bytes,off_t offset) {
  while (bytes) {
    const ssize_t put=::pwrite(fd,data,bytes,offset);
    if (put<0 && errno==EINTR) continue;
    if (put<=0) bad(std::string("row write failed: ")+std::strerror(errno));
    data+=put; bytes-=size_t(put); offset+=put;
  }
}
} // namespace

TieredQMemoryRows::TieredQMemoryRows(std::vector<uint8_t> data,size_t row_bytes)
    : data_(std::move(data)),row_bytes_(row_bytes),rows_(row_bytes ? data_.size()/row_bytes : 0) {
  if (!row_bytes_ || data_.size()%row_bytes_) bad("row data is not a whole number of rows");
}
void TieredQMemoryRows::read(const uint32_t* ids,size_t count,uint8_t* out) const {
  for (size_t i=0;i<count;++i) {
    check_row_id(ids[i],rows_);
    std::memcpy(out+i*row_bytes_,data_.data()+size_t(ids[i])*row_bytes_,row_bytes_);
  }
}

TieredQFileRows::TieredQFileRows(const std::string& path) {
  fd_=::open(path.c_str(),O_RDONLY|O_CLOEXEC);
  if (fd_<0) bad(system_error("cannot open row file",path));
  try {
    std::vector<uint8_t> bytes(row_file_header_bytes);
    read_exact(fd_,bytes.data(),bytes.size(),0);
    tieredq_detail::Reader header{bytes};
    if (header.u32()!=row_file_magic) bad("not a TieredQ row file: "+path);
    if (header.u32()!=format) bad("unsupported row file format");
    row_bytes_=size_t(header.u64()); rows_=size_t(header.u64());
    struct stat info{};
    if (::fstat(fd_,&info)) bad(system_error("cannot stat row file",path));
    if (!row_bytes_ || rows_>(size_t(std::numeric_limits<off_t>::max())-data_alignment)/row_bytes_ ||
        size_t(info.st_size)!=data_alignment+rows_*row_bytes_)
      bad("row file size does not match its header");
#ifdef POSIX_FADV_RANDOM
    // Candidates are scattered ids: readahead would only evict useful pages.
    ::posix_fadvise(fd_,0,0,POSIX_FADV_RANDOM);
#endif
  } catch (...) { ::close(fd_); throw; }
}
TieredQFileRows::~TieredQFileRows() { if (fd_>=0) ::close(fd_); }
void TieredQFileRows::read(const uint32_t* ids,size_t count,uint8_t* out) const {
  for (size_t i=0;i<count;++i) {
    check_row_id(ids[i],rows_);
    read_exact(fd_,out+i*row_bytes_,row_bytes_,off_t(data_alignment+size_t(ids[i])*row_bytes_));
  }
}

TieredQRowFileWriter::TieredQRowFileWriter(const std::string& path,size_t row_bytes) : row_bytes_(row_bytes) {
  if (!row_bytes_) bad("row file needs a positive row size");
  fd_=::open(path.c_str(),O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC,0644);
  if (fd_<0) bad(system_error("cannot create row file",path));
}
TieredQRowFileWriter::~TieredQRowFileWriter() { if (fd_>=0) ::close(fd_); }
void TieredQRowFileWriter::append(const uint8_t* rows,size_t count) {
  if (fd_<0) bad("row file is finished");
  if (count && !rows) bad("null row data");
  write_exact(fd_,rows,product(count,row_bytes_),off_t(TieredQFileRows::data_alignment+rows_*row_bytes_));
  rows_+=count;
}
// The header goes last, so an unfinished file is never mistaken for a complete one.
void TieredQRowFileWriter::finish() {
  if (fd_<0) bad("row file is finished");
  tieredq_detail::Writer header;
  header.u32(row_file_magic); header.u32(TieredQFileRows::format); header.u64(row_bytes_); header.u64(rows_);
  header.data.resize(TieredQFileRows::data_alignment,0);
  write_exact(fd_,header.data.data(),header.data.size(),0);
  const int closed=::close(fd_);
  fd_=-1;
  if (closed) bad(std::string("row file close failed: ")+std::strerror(errno));
}

TieredQNestedIVFIndex::TieredQNestedIVFIndex(TieredQIVFIndex index,TieredQRefinement refinement,TieredQTierRows tier_rows)
    : index_(std::move(index)),refinement_(std::move(refinement)),tier_rows_(tier_rows) {
  if (refinement_.prefix().model_id()!=index_.codec().model_id()) bad("refinement was trained on another prefix model");
  if (tier_rows_!=TieredQTierRows::prefix_and_suffix && tier_rows_!=TieredQTierRows::suffix) bad("invalid tier rows");
  if (tier_rows_==TieredQTierRows::suffix) documents_=index_.id_documents();
}

size_t TieredQNestedIVFIndex::tier_bytes_per_vector() const {
  const size_t prefix=tier_rows_==TieredQTierRows::prefix_and_suffix ? index_.codec().bytes_per_vector() : 0;
  return prefix+refinement_.bytes_per_vector();
}

void TieredQNestedIVFIndex::encode_tier(const float* rows,size_t first,size_t count,uint8_t* out) const {
  if (first>size() || count>size()-first) bad("tier rows out of range");
  if (!count) return;
  if (!rows || !out) bad("null tier buffer");
  const size_t d=index_.codec().config().dimensions,prefix_stride=index_.codec().bytes_per_vector();
  const size_t suffix_stride=refinement_.bytes_per_vector(),tier_stride=tier_bytes_per_vector();
  const bool inline_prefix=tier_rows_==TieredQTierRows::prefix_and_suffix;
  // Prefix rows come from the lists, so the suffix refines exactly the codes the index scans.
  const auto all_documents=inline_prefix ? index_.id_documents() : std::vector<uint32_t>();
  const uint32_t* documents=(inline_prefix ? all_documents : documents_).data()+first;
  std::vector<uint8_t> prefix_rows,suffix_rows;
  for (size_t done=0;done<count;done+=encode_chunk_rows) {
    const size_t n=std::min(encode_chunk_rows,count-done);
    prefix_rows.resize(n*prefix_stride); suffix_rows.resize(n*suffix_stride);
    index_.document_rows(documents+done,n,prefix_rows.data());
    refinement_.encode(rows+done*d,n,prefix_rows.data(),suffix_rows.data());
    for (size_t i=0;i<n;++i) {
      uint8_t* row=out+(done+i)*tier_stride;
      if (inline_prefix) { std::memcpy(row,prefix_rows.data()+i*prefix_stride,prefix_stride); row+=prefix_stride; }
      std::memcpy(row,suffix_rows.data()+i*suffix_stride,suffix_stride);
    }
  }
}

void TieredQNestedIVFIndex::attach_tier(std::shared_ptr<const TieredQRowSource> tier) {
  if (!tier) bad("null tier source");
  if (tier->row_bytes()!=tier_bytes_per_vector()) bad("tier row size does not match the index");
  if (tier->rows()!=size()) bad("tier row count does not match the index");
  tier_=std::move(tier);
}

void TieredQNestedIVFIndex::candidate_rows(const uint32_t* candidates,size_t found,uint8_t* prefix_rows,
                                           uint8_t* suffix_rows) const {
  const size_t prefix_stride=index_.codec().bytes_per_vector(),suffix_stride=refinement_.bytes_per_vector();
  if (tier_rows_==TieredQTierRows::suffix) {
    thread_local std::vector<uint32_t> documents;
    documents.resize(found);
    for (size_t i=0;i<found;++i) documents[i]=documents_[candidates[i]];
    index_.document_rows(documents.data(),found,prefix_rows);
    tier_->read(candidates,found,suffix_rows);
    return;
  }
  thread_local std::vector<uint8_t> tier_rows;
  const size_t tier_stride=tier_bytes_per_vector();
  tier_rows.resize(found*tier_stride);
  tier_->read(candidates,found,tier_rows.data());
  for (size_t i=0;i<found;++i) {
    std::memcpy(prefix_rows+i*prefix_stride,tier_rows.data()+i*tier_stride,prefix_stride);
    std::memcpy(suffix_rows+i*suffix_stride,tier_rows.data()+i*tier_stride+prefix_stride,suffix_stride);
  }
}

size_t TieredQNestedIVFIndex::rescore(const float* query,TieredQQuery weights,const uint32_t* candidates,size_t found,
                                      size_t k,float* scores,uint32_t* ids) const {
  if (!found) return 0;
  thread_local std::vector<uint32_t> order;
  thread_local std::vector<uint8_t> prefix_rows,suffix_rows;
  thread_local std::vector<float> refined;
  order.resize(found); refined.resize(found);
  prefix_rows.resize(found*index_.codec().bytes_per_vector()); suffix_rows.resize(found*refinement_.bytes_per_vector());
  candidate_rows(candidates,found,prefix_rows.data(),suffix_rows.data());
  refinement_.score(prefix_rows.data(),suffix_rows.data(),found,refinement_.query(query,std::move(weights)),refined.data());
  const size_t kept=std::min(k,found);
  std::iota(order.begin(),order.end(),uint32_t(0));
  std::partial_sort(order.begin(),order.begin()+kept,order.end(),[&](uint32_t a,uint32_t b) {
    return refined[a]!=refined[b] ? refined[a]>refined[b] : candidates[a]<candidates[b];
  });
  const auto [scale,offset]=index_.codec().score_map(query);
  for (size_t i=0;i<kept;++i) { scores[i]=float(scale*refined[order[i]]+offset); ids[i]=candidates[order[i]]; }
  return kept;
}

std::vector<TieredQHit> TieredQNestedIVFIndex::search(const float* query,size_t k,size_t nprobe,size_t candidates,
                                                      TieredQPruning pruning) const {
  if (!tier_) bad("no tier rows attached");
  if (candidates<k) bad("rescoring needs at least k candidates");
  if (!k || !size()) return {};
  if (!query) bad("null query");
  auto weights=index_.codec().query_weights(query);
  const auto hits=index_.search(query,weights,candidates,nprobe,pruning);
  std::vector<uint32_t> found(hits.size());
  for (size_t i=0;i<hits.size();++i) found[i]=hits[i].id;
  std::vector<float> scores(std::min(k,found.size()));
  std::vector<uint32_t> ids(scores.size());
  const size_t kept=rescore(query,std::move(weights),found.data(),found.size(),k,scores.data(),ids.data());
  std::vector<TieredQHit> out(kept);
  for (size_t i=0;i<kept;++i) out[i]={scores[i],ids[i]};
  return out;
}

void TieredQNestedIVFIndex::search_batch(const float* queries,size_t rows,size_t k,size_t nprobe,size_t candidates,
                                         float* scores,uint32_t* ids,TieredQPruning pruning) const {
  if (!tier_) bad("no tier rows attached");
  if (candidates<k) bad("rescoring needs at least k candidates");
  if (!rows || !k || !size()) return;
  if (!queries || !scores || !ids) bad("null search buffer");
  const TieredQ& model=index_.codec();
  const size_t d=model.config().dimensions,width=std::min(k,size()),depth=std::min(candidates,size());
  auto weights=model.query_weights(queries,rows);
  std::vector<float> found_scores(product(rows,depth));
  std::vector<uint32_t> found_ids(product(rows,depth));
  index_.search_batch(queries,weights.data(),rows,candidates,nprobe,found_scores.data(),found_ids.data(),pruning);
  #pragma omp parallel for schedule(dynamic)
  for (size_t q=0;q<rows;++q) {
    const uint32_t* row_ids=found_ids.data()+q*depth;
    const size_t found=size_t(std::find(row_ids,row_ids+depth,padding_id)-row_ids);
    float* out_scores=scores+q*width; uint32_t* out_ids=ids+q*width;
    const size_t kept=rescore(queries+q*d,std::move(weights[q]),row_ids,found,k,out_scores,out_ids);
    std::fill(out_scores+kept,out_scores+width,-std::numeric_limits<float>::infinity());
    std::fill(out_ids+kept,out_ids+width,padding_id);
  }
}

std::vector<uint8_t> TieredQNestedIVFIndex::serialize() const {
  tieredq_detail::Writer writer;
  writer.u32(nested_payload_version);
  writer.u32(uint32_t(tier_rows_));
  const auto index=index_.serialize(),refinement=refinement_.serialize();
  writer.u64(index.size()); writer.data.insert(writer.data.end(),index.begin(),index.end());
  writer.u64(refinement.size()); writer.data.insert(writer.data.end(),refinement.begin(),refinement.end());
  return std::move(writer.data);
}

TieredQNestedIVFIndex TieredQNestedIVFIndex::deserialize(const std::vector<uint8_t>& bytes) {
  tieredq_detail::Reader reader{bytes};
  if (reader.u32()!=nested_payload_version) bad("unsupported nested index version");
  const auto tier_rows=TieredQTierRows(reader.u32());
  auto section=[&] {
    const uint64_t length=reader.u64();
    if (length>reader.remaining()) bad("truncated nested index");
    std::vector<uint8_t> data(length);
    reader.bytes(data.data(),data.size());
    return data;
  };
  auto index=TieredQIVFIndex::deserialize(section());
  auto refinement=TieredQRefinement::deserialize(section(),index.codec());
  if (reader.remaining()) bad("trailing nested index bytes");
  return TieredQNestedIVFIndex(std::move(index),std::move(refinement),tier_rows);
}
} // namespace dv
