// Checked little-endian model/index containers with checksums and atomic installation.
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace dv::tieredq_detail {
constexpr uint32_t codec_file_kind=1,exact_index_file_kind=2,fast_index_file_kind=3,ivf_index_file_kind=4;
inline void bad(const std::string& message) { throw std::runtime_error("TieredQ: "+message); }
inline size_t product(size_t a,size_t b) {
  if (b && a>std::numeric_limits<size_t>::max()/b) bad("size overflow");
  return a*b;
}
// Version 1: little-endian integers and IEEE float32, length-prefixed arrays.
struct Writer {
  std::vector<uint8_t> data;
  void u32(uint32_t x) { for (int i=0;i<4;++i) data.push_back(uint8_t(x>>(8*i))); }
  void u64(uint64_t x) { u32(uint32_t(x)); u32(uint32_t(x>>32)); }
  void f32(float x) { u32(std::bit_cast<uint32_t>(x)); }
  void floats(const std::vector<float>& x) { u64(x.size()); for (float v:x) f32(v); }
  void halves(const std::vector<uint16_t>& x) {
    u64(x.size()); for (uint16_t v:x) { data.push_back(uint8_t(v)); data.push_back(uint8_t(v>>8)); }
  }
};
struct Reader {
  const std::vector<uint8_t>& data; size_t pos=0;
  uint32_t u32() {
    if (data.size()-pos<4) bad("truncated metadata");
    uint32_t x=0; for (int i=0;i<4;++i) x |= uint32_t(data[pos++])<<(8*i); return x;
  }
  uint64_t u64() { uint64_t lo=u32(); return lo|(uint64_t(u32())<<32); }
  float f32() { float x=std::bit_cast<float>(u32()); if (!std::isfinite(x)) bad("nonfinite model"); return x; }
  std::vector<float> floats() {
    uint64_t n=u64(); if (n>(data.size()-pos)/4) bad("invalid array length");
    std::vector<float> x(static_cast<size_t>(n)); for (auto& v:x) v=f32(); return x;
  }
  std::vector<uint16_t> halves() {
    uint64_t n=u64(); if (n>(data.size()-pos)/2) bad("invalid array length");
    std::vector<uint16_t> x(static_cast<size_t>(n));
    for (auto& v:x) { v=uint16_t(data[pos]|uint16_t(data[pos+1])<<8); pos+=2; }
    return x;
  }
  size_t remaining() const { return data.size()-pos; }
  void bytes(void* out,size_t n) {
    if (n>remaining()) bad("truncated metadata");
    if (n) std::memcpy(out,data.data()+pos,n);
    pos+=n;
  }
};
constexpr uint64_t checksum_seed=14695981039346656037ULL;
inline uint64_t checksum_update(uint64_t hash, const uint8_t* p, size_t n) {
  for (size_t i=0;i<n;++i) { hash ^= p[i]; hash *= 1099511628211ULL; } return hash;
}
inline uint64_t checksum(const uint8_t* p, size_t n) { return checksum_update(checksum_seed,p,n); }
// Bulk little-endian float32 array into preallocated values; rejects nonfinite entries.
template<class R> void read_f32_array(R& reader,float* values,size_t count) {
  reader.bytes(values,product(count,sizeof(float)));
  if constexpr (std::endian::native==std::endian::big)
    for (size_t i=0;i<count;++i) {
      const uint32_t x=std::bit_cast<uint32_t>(values[i]);
      values[i]=std::bit_cast<float>((x>>24)|((x>>8)&0xff00u)|((x<<8)&0xff0000u)|(x<<24));
    }
  for (size_t i=0;i<count;++i) if (!std::isfinite(values[i])) bad("nonfinite model");
}
// Header: magic[8], version, kind, model offset/length, payload offset/length,
// row count, model checksum, payload checksum (80 bytes).
inline void save_file(const std::string& path, const std::vector<uint8_t>& model,
               const std::vector<uint8_t>& payload, uint64_t rows, uint32_t kind) {
  Writer w; for (char c: std::array<char,8>{'T','I','E','R','E','D','Q','\0'}) w.data.push_back(uint8_t(c));
  // The container version and codec recipe have independent version numbers.
  w.u32(1); w.u32(kind); w.u64(80); w.u64(model.size()); w.u64(80+model.size()); w.u64(payload.size());
  w.u64(rows); w.u64(checksum(model.data(),model.size())); w.u64(checksum(payload.data(),payload.size())); w.u64(0);
  static std::atomic<uint64_t> sequence{0};
  const std::string temporary=path+".tmp."+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+"."+std::to_string(sequence++);
  struct Cleanup {
    std::string path;
    ~Cleanup() { std::error_code ec; std::filesystem::remove(path,ec); }
  } cleanup{temporary};
  std::ofstream out(temporary,std::ios::binary|std::ios::trunc); if (!out) bad("cannot create "+temporary);
  out.write(reinterpret_cast<const char*>(w.data.data()),w.data.size());
  out.write(reinterpret_cast<const char*>(model.data()),model.size());
  out.write(reinterpret_cast<const char*>(payload.data()),payload.size()); out.close();
  if (!out) bad("write failed: "+path);
  std::error_code error; std::filesystem::rename(temporary,path,error);
  if (error) bad("cannot install "+path+": "+error.message());
}
constexpr size_t header_bytes=80;
struct Header { uint64_t model_length,payload_length,rows,model_hash,payload_hash; uint32_t kind; };
inline Header read_header(std::ifstream& in, size_t file_bytes, uint32_t kind) {
  if (file_bytes<header_bytes) bad("truncated header");
  std::vector<uint8_t> data(header_bytes);
  in.read(reinterpret_cast<char*>(data.data()),data.size()); if (!in) bad("read failed");
  if (std::memcmp(data.data(),"TIEREDQ\0",8)) bad("invalid magic");
  Reader r{data,8}; if (r.u32()!=1) bad("unsupported format version");
  const uint32_t actual_kind=r.u32(); if (kind && actual_kind!=kind) bad("wrong file kind");
  uint64_t mo=r.u64(),ml=r.u64(),po=r.u64(),pl=r.u64(),rows=r.u64(),mh=r.u64(),ph=r.u64();
  if (r.u64()!=0 || mo!=header_bytes || ml>file_bytes-header_bytes || po!=header_bytes+ml || pl!=file_bytes-po) bad("invalid file offsets");
  if (actual_kind==codec_file_kind && (rows || pl)) bad("codec file contains payload");
  return {ml,pl,rows,mh,ph,actual_kind};
}
// Checked header and model; the stream is left at the payload for in-place parsing.
struct OpenFile { std::ifstream in; Header header; std::vector<uint8_t> model; };
inline OpenFile open_file(const std::string& path, uint32_t kind) {
  OpenFile file{std::ifstream(path,std::ios::binary|std::ios::ate),{},{}};
  if (!file.in) bad("cannot open "+path);
  const auto end=file.in.tellg(); file.in.seekg(0);
  file.header=read_header(file.in,static_cast<size_t>(end),kind);
  file.model.resize(file.header.model_length);
  file.in.read(reinterpret_cast<char*>(file.model.data()),file.model.size()); if (!file.in) bad("read failed");
  if (file.header.model_hash!=checksum(file.model.data(),file.model.size())) bad("checksum mismatch");
  return file;
}
// Buffered payload reader that checksums every byte it consumes; finish() verifies the whole payload.
struct StreamReader {
  static constexpr size_t buffer_bytes=size_t(1)<<20;
  std::ifstream& in; size_t left; uint64_t hash=checksum_seed;
  std::vector<uint8_t> buffer; size_t pos=0,end=0;
  StreamReader(std::ifstream& stream,size_t length) : in(stream),left(length),buffer(std::min(length,buffer_bytes)) {}
  size_t remaining() const { return end-pos+left; }
  void bytes(void* out,size_t n) {
    if (n>remaining()) bad("truncated metadata");
    auto* target=static_cast<uint8_t*>(out);
    const size_t buffered=std::min(n,end-pos);
    if (buffered) std::memcpy(target,buffer.data()+pos,buffered);
    pos+=buffered; target+=buffered; n-=buffered;
    if (n>=buffer.size()) { pull(target,n); return; }
    if (!n) return;
    refill(); std::memcpy(target,buffer.data(),n); pos=n;
  }
  uint32_t u32() {
    std::array<uint8_t,4> b; bytes(b.data(),b.size());
    return uint32_t(b[0])|uint32_t(b[1])<<8|uint32_t(b[2])<<16|uint32_t(b[3])<<24;
  }
  uint64_t u64() { uint64_t lo=u32(); return lo|(uint64_t(u32())<<32); }
  // Consumes any unread bytes so corruption reports as a checksum failure before parse errors.
  bool verify(uint64_t expected) {
    pos=end;
    while (left) refill();
    pos=end;
    return hash==expected;
  }
 private:
  void pull(uint8_t* target,size_t n) {
    in.read(reinterpret_cast<char*>(target),static_cast<std::streamsize>(n)); if (!in) bad("read failed");
    hash=checksum_update(hash,target,n); left-=n;
  }
  void refill() { const size_t n=std::min(left,buffer.size()); pull(buffer.data(),n); pos=0; end=n; }
};
struct File { std::vector<uint8_t> model,payload; uint64_t rows; uint32_t kind; };
inline File load_file(const std::string& path, uint32_t kind) {
  auto file=open_file(path,kind);
  std::vector<uint8_t> payload(file.header.payload_length);
  file.in.read(reinterpret_cast<char*>(payload.data()),payload.size()); if (!file.in) bad("read failed");
  if (file.header.payload_hash!=checksum(payload.data(),payload.size())) bad("checksum mismatch");
  return {std::move(file.model),std::move(payload),file.header.rows,file.header.kind};
}
} // namespace dv::tieredq_detail
