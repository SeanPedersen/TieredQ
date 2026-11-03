// Nested IVF serving (#23): prefix scan, then prefix plus suffix rescoring of the k' candidates, plus the RAM and
// file row sources the tier is read from.
#include "tieredq/tieredq_nested.h"
#include "tieredq_io.h"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#ifdef DV_TIEREDQ_URING
#include <liburing.h>
#endif
#include <limits>
#include <memory>
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
// One planned positioned read: `bytes` from `offset` into `into`, of which the first `needed` must arrive (direct
// reads may stop short at the end of the file).
struct RowRead { size_t offset, bytes, needed; uint8_t* into; };
constexpr size_t max_queue_depth=4096;

// Completes a planned read from `got` bytes on, one pread at a time.
void read_planned(int fd,const RowRead& row,size_t got) {
  while (got<row.needed) {
    const ssize_t n=::pread(fd,row.into+got,row.bytes-got,off_t(row.offset+got));
    if (n<0 && errno==EINTR) continue;
    if (n<=0) bad(n<0 ? std::string("row read failed: ")+std::strerror(errno) : "truncated row file");
    got+=size_t(n);
  }
}

#ifdef DV_TIEREDQ_URING
// One ring per thread: read() runs concurrently on search workers, and a ring is not thread-safe.
struct ThreadRing {
  io_uring ring{};
  unsigned entries=0;
  ~ThreadRing() { if (entries) io_uring_queue_exit(&ring); }
  io_uring& at_least(unsigned depth) {
    if (entries>=depth) return ring;
    if (entries) { io_uring_queue_exit(&ring); entries=0; }
    if (const int error=io_uring_queue_init(depth,&ring,0)) bad(std::string("io_uring setup failed: ")+std::strerror(-error));
    entries=depth;
    return ring;
  }
};
// Keeps up to depth reads in flight, refilling as each completes; short reads finish with pread.
void read_concurrent(int fd,const std::vector<RowRead>& plan,size_t depth) {
  thread_local ThreadRing thread_ring;
  const unsigned window=unsigned(std::min(depth,plan.size()));
  io_uring& ring=thread_ring.at_least(window);
  size_t queued=0,done=0;
  while (done<plan.size()) {
    unsigned prepared=0;
    for (;queued<plan.size() && queued-done<window;++queued,++prepared) {
      io_uring_sqe* sqe=io_uring_get_sqe(&ring);
      if (!sqe) break;
      const auto& row=plan[queued];
      io_uring_prep_read(sqe,fd,row.into,unsigned(row.bytes),row.offset);
      io_uring_sqe_set_data64(sqe,queued);
    }
    if (prepared && io_uring_submit(&ring)<0) bad("io_uring submit failed");
    io_uring_cqe* cqe=nullptr;
    int error=io_uring_wait_cqe(&ring,&cqe);
    while (error==-EINTR) error=io_uring_wait_cqe(&ring,&cqe);
    if (error) bad(std::string("io_uring wait failed: ")+std::strerror(-error));
    const auto& row=plan[size_t(io_uring_cqe_get_data64(cqe))];
    const int result=cqe->res;
    io_uring_cqe_seen(&ring,cqe);
    if (result<0 && result!=-EINTR && result!=-EAGAIN) bad(std::string("row read failed: ")+std::strerror(-result));
    read_planned(fd,row,result>0 ? size_t(result) : 0);
    ++done;
  }
}
#endif
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

// Direct I/O: O_DIRECT where the platform has it, F_NOCACHE on macOS.
static int open_rows(const std::string& path,TieredQFileReads reads) {
  int flags=O_RDONLY|O_CLOEXEC;
#ifdef O_DIRECT
  if (reads==TieredQFileReads::direct) flags|=O_DIRECT;
#endif
  const int fd=::open(path.c_str(),flags);
  if (fd<0) bad(system_error(reads==TieredQFileReads::direct ? "cannot open row file for direct reads" : "cannot open row file",path));
#if !defined(O_DIRECT) && defined(F_NOCACHE)
  if (reads==TieredQFileReads::direct && ::fcntl(fd,F_NOCACHE,1)) { ::close(fd); bad(system_error("cannot disable caching of",path)); }
#elif !defined(O_DIRECT)
  if (reads==TieredQFileReads::direct) { ::close(fd); bad("direct reads are not supported on this platform"); }
#endif
  return fd;
}

TieredQFileRows::TieredQFileRows(const std::string& path,TieredQFileReads reads,size_t queue_depth)
    : reads_(reads),queue_depth_(queue_depth) {
  if (!queue_depth_ || queue_depth_>max_queue_depth) bad("queue depth must be 1 to "+std::to_string(max_queue_depth));
  if (reads_!=TieredQFileReads::cached && reads_!=TieredQFileReads::direct) bad("invalid file reads");
  // The header is read through the cache; direct reads then use their own descriptor.
  fd_=open_rows(path,TieredQFileReads::cached);
  try {
    std::vector<uint8_t> bytes(row_file_header_bytes);
    read_exact(fd_,bytes.data(),bytes.size(),0);
    tieredq_detail::Reader header{bytes};
    if (header.u32()!=row_file_magic) bad("not a TieredQ row file: "+path);
    if (header.u32()!=format) bad("unsupported row file format");
    row_bytes_=size_t(header.u64()); rows_=size_t(header.u64());
    struct stat info{};
    if (::fstat(fd_,&info)) bad(system_error("cannot stat row file",path));
    if (!row_bytes_ || rows_>(size_t(std::numeric_limits<off_t>::max())-2*data_alignment)/row_bytes_ ||
        size_t(info.st_size)!=data_alignment+rows_*row_bytes_)
      bad("row file size does not match its header");
    file_bytes_=size_t(info.st_size);
    if (reads_==TieredQFileReads::direct) { ::close(fd_); fd_=-1; fd_=open_rows(path,reads_); }
#ifdef POSIX_FADV_RANDOM
    // Candidates are scattered ids: readahead would only evict useful pages.
    else ::posix_fadvise(fd_,0,0,POSIX_FADV_RANDOM);
#endif
  } catch (...) { if (fd_>=0) ::close(fd_); throw; }
}
TieredQFileRows::~TieredQFileRows() { if (fd_>=0) ::close(fd_); }
bool TieredQFileRows::concurrent_reads_available() {
#ifdef DV_TIEREDQ_URING
  return true;
#else
  return false;
#endif
}

// Cached rows read straight into out. Direct reads need block-aligned offsets, lengths and buffers: each candidate
// reads the pages covering its row (the last may end past the file) into its own slot, then copies the row out.
void TieredQFileRows::read(const uint32_t* ids,size_t count,uint8_t* out) const {
  if (!count) return;
  const bool direct=reads_==TieredQFileReads::direct;
  const size_t span=direct ? (row_bytes_+2*data_alignment-2)/data_alignment*data_alignment : 0;
  const size_t file_end=(file_bytes_+data_alignment-1)/data_alignment*data_alignment;
  struct Aligned { void operator()(uint8_t* p) const { std::free(p); } };
  thread_local std::unique_ptr<uint8_t,Aligned> pages;
  thread_local size_t page_bytes=0;
  if (direct && page_bytes<count*span) {
    pages.reset(static_cast<uint8_t*>(std::aligned_alloc(data_alignment,count*span)));
    if (!pages) { page_bytes=0; bad("cannot allocate direct read buffer"); }
    page_bytes=count*span;
  }
  thread_local std::vector<RowRead> plan;
  plan.resize(count);
  for (size_t i=0;i<count;++i) {
    check_row_id(ids[i],rows_);
    const size_t offset=data_alignment+size_t(ids[i])*row_bytes_;
    if (!direct) { plan[i]={offset,row_bytes_,row_bytes_,out+i*row_bytes_}; continue; }
    const size_t first=offset/data_alignment*data_alignment;
    const size_t last=std::min((offset+row_bytes_+data_alignment-1)/data_alignment*data_alignment,file_end);
    plan[i]={first,last-first,offset+row_bytes_-first,pages.get()+i*span};
  }
#ifdef DV_TIEREDQ_URING
  if (queue_depth_>1) read_concurrent(fd_,plan,queue_depth_); else
#endif
  for (const auto& row:plan) read_planned(fd_,row,0);
  if (direct) for (size_t i=0;i<count;++i)
    std::memcpy(out+i*row_bytes_,plan[i].into+(data_alignment+size_t(ids[i])*row_bytes_-plan[i].offset),row_bytes_);
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

size_t TieredQNestedIVFIndex::rescore(const float* query,const TieredQRefinementQuery& refined_query,const uint32_t* candidates,
                                      size_t found,size_t k,float* scores,uint32_t* ids) const {
  if (!found) return 0;
  thread_local std::vector<uint32_t> order;
  thread_local std::vector<uint8_t> prefix_rows,suffix_rows;
  thread_local std::vector<float> refined;
  order.resize(found); refined.resize(found);
  prefix_rows.resize(found*index_.codec().bytes_per_vector()); suffix_rows.resize(found*refinement_.bytes_per_vector());
  candidate_rows(candidates,found,prefix_rows.data(),suffix_rows.data());
  refinement_.score(prefix_rows.data(),suffix_rows.data(),found,refined_query,refined.data());
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
  const size_t kept=rescore(query,refinement_.query(query,std::move(weights)),found.data(),found.size(),k,scores.data(),ids.data());
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
  // Projected together, the queries share each pass over the suffix basis.
  const auto refined_queries=refinement_.query(queries,rows,std::move(weights));
  #pragma omp parallel for schedule(dynamic)
  for (size_t q=0;q<rows;++q) {
    const uint32_t* row_ids=found_ids.data()+q*depth;
    const size_t found=size_t(std::find(row_ids,row_ids+depth,padding_id)-row_ids);
    float* out_scores=scores+q*width; uint32_t* out_ids=ids+q*width;
    const size_t kept=rescore(queries+q*d,refined_queries[q],row_ids,found,k,out_scores,out_ids);
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
