// Checked little-endian model/index containers with checksums and atomic installation.
#pragma once
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
constexpr uint32_t codec_file_kind=1,exact_index_file_kind=2,fast_index_file_kind=3;
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
};
inline uint64_t checksum(const uint8_t* p, size_t n) {
  uint64_t hash=14695981039346656037ULL;
  for (size_t i=0;i<n;++i) { hash ^= p[i]; hash *= 1099511628211ULL; } return hash;
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
struct File { std::vector<uint8_t> model,payload; uint64_t rows; uint32_t kind; };
inline File load_file(const std::string& path, uint32_t kind) {
  std::ifstream in(path,std::ios::binary|std::ios::ate); if (!in) bad("cannot open "+path);
  const auto end=in.tellg(); if (end<80) bad("truncated header");
  std::vector<uint8_t> data(static_cast<size_t>(end)); in.seekg(0);
  in.read(reinterpret_cast<char*>(data.data()),data.size()); if (!in) bad("read failed");
  if (std::memcmp(data.data(),"TIEREDQ\0",8)) bad("invalid magic");
  Reader r{data,8}; if (r.u32()!=1) bad("unsupported format version");
  const uint32_t actual_kind=r.u32(); if (kind && actual_kind!=kind) bad("wrong file kind");
  uint64_t mo=r.u64(),ml=r.u64(),po=r.u64(),pl=r.u64(),rows=r.u64(),mh=r.u64(),ph=r.u64();
  if (r.u64()!=0 || mo!=80 || ml>data.size()-80 || po!=80+ml || pl!=data.size()-po) bad("invalid file offsets");
  if (mh!=checksum(data.data()+mo,ml) || ph!=checksum(data.data()+po,pl)) bad("checksum mismatch");
  if (actual_kind==codec_file_kind && (rows || pl)) bad("codec file contains payload");
  return {{data.begin()+mo,data.begin()+po},{data.begin()+po,data.end()},rows,actual_kind};
}
} // namespace dv::tieredq_detail
