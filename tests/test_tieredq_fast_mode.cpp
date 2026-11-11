// Fast-only copy-on-write ownership, moved/empty index persistence and chunked bulk adds.
#include "tieredq/tieredq.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <random>
#include <stdexcept>
#include <unistd.h>
#include <type_traits>
static_assert(!std::is_constructible_v<tq::TieredQIndex,tq::TieredQ,bool>);
namespace {
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
void ownership_test(uint32_t bits) {
  constexpr uint32_t dimensions=8,centroids=4;
  constexpr size_t training_rows=128,rows=257,append_rows=17,k=10;
  std::mt19937 random(19);std::normal_distribution<float> normal;
  std::vector<float> training(training_rows*dimensions),data((rows+append_rows)*dimensions);
  for (auto& value:training) value=normal(random);
  for (auto& value:data) value=normal(random);
  auto codec=tq::TieredQ::train({dimensions,bits,centroids,false},training.data(),training_rows);
  tq::TieredQIndex index(codec);
  try {
    tq::TieredQIndex invalid(codec,static_cast<tq::TieredQIndexMode>(255));
    throw std::runtime_error("invalid mode accepted");
  } catch (const std::invalid_argument&) {}
  require(index.mode()==tq::TieredQIndexMode::fast && index.storage_bytes()==0,"empty fast-only storage is not empty");
  const auto path=std::filesystem::temp_directory_path()/("tieredq-fast-only-"+std::to_string(getpid())+"-"+std::to_string(bits));
  index.save(path.string());
  auto empty=tq::TieredQIndex::load(path.string());
  require(empty.mode()==tq::TieredQIndexMode::fast && empty.size()==0,"empty fast-only mode not persisted");
  empty.add(data.data(),rows);
  index.add(data.data(),rows);
  require(index.uses_scalar_fast_scan()!=tq::TieredQIndex::native_fast_scan_available(),"fast backend diagnostic disagrees with native capability");
  auto before=index.search(data.data(),k);
  auto copy=index;
  copy.add(data.data()+rows*dimensions,append_rows);
  require(index.size()==rows && copy.size()==rows+append_rows,"fast-only copy shares row count");
  const auto after=index.search(data.data(),k);
  for (size_t i=0;i<k;++i)
    require(after[i].id==before[i].id && after[i].score==before[i].score,"append changed a sibling fast-only index");
  auto moved=std::move(copy);
  require(moved.mode()==tq::TieredQIndexMode::fast && moved.size()==rows+append_rows && copy.size()==0,"move lost fast-only state");
  moved.save(path.string());
  auto loaded=tq::TieredQIndex::load(path.string());
  const auto moved_hits=moved.search(data.data(),k);
  const auto loaded_hits=loaded.search(data.data(),k);
  for (size_t i=0;i<k;++i)
    require(loaded_hits[i].id==moved_hits[i].id && loaded_hits[i].score==moved_hits[i].score,
        "reload changes fast-only scores");
  loaded.add(data.data(),append_rows);
  moved.add(data.data(),append_rows);
  const auto appended=moved.search(data.data(),k);
  const auto reloaded_append=loaded.search(data.data(),k);
  for (size_t i=0;i<k;++i)
    require(reloaded_append[i].id==appended[i].id && reloaded_append[i].score==appended[i].score,
        "append after reload changes fast-only scores");
  require(loaded.mode()==tq::TieredQIndexMode::fast && loaded.size()==rows+2*append_rows,"loaded fast-only index cannot append");
  std::filesystem::remove(path);
}
// Payload larger than the 1 MiB load buffer exercises buffered, direct and checksum-only reads.
void streamed_load_test() {
  constexpr uint32_t dimensions=64,bits=256,centroids=16;
  constexpr size_t training_rows=2048,rows=100000,k=10;
  std::mt19937 random(23);std::normal_distribution<float> normal;
  std::vector<float> data(rows*dimensions);
  for (auto& value:data) value=normal(random);
  auto codec=tq::TieredQ::train({dimensions,bits,centroids,false},data.data(),training_rows);
  for (const auto mode:{tq::TieredQIndexMode::fast,tq::TieredQIndexMode::full}) {
    tq::TieredQIndex index(codec,mode);
    index.add(data.data(),rows);
    const auto path=std::filesystem::temp_directory_path()/("tieredq-streamed-"+std::to_string(getpid()));
    index.save(path.string());
    require(std::filesystem::file_size(path)>(size_t(1)<<21),"streamed load test payload fits one buffer");
    auto loaded=tq::TieredQIndex::load(path.string());
    require(loaded.storage_bytes()==index.storage_bytes() && loaded.size()==rows,"streamed load changes storage");
    for (size_t query=0;query<rows;query+=rows/8) {
      const auto expected=index.search(data.data()+query*dimensions,k),actual=loaded.search(data.data()+query*dimensions,k);
      for (size_t i=0;i<k;++i)
        require(actual[i].id==expected[i].id && actual[i].score==expected[i].score,"streamed load changes scores");
    }
    {
      std::fstream file(path,std::ios::binary|std::ios::in|std::ios::out);
      file.seekg(std::streamoff(std::filesystem::file_size(path)/2));
      const char byte=char(file.get()^1);
      file.seekp(std::streamoff(std::filesystem::file_size(path)/2));file.put(byte);
    }
    bool rejected=false;
    try { tq::TieredQIndex::load(path.string()); }
    catch (const std::runtime_error& error) { rejected=std::string(error.what()).find("checksum mismatch")!=std::string::npos; }
    require(rejected,"streamed load accepts a corrupted payload");
    std::filesystem::remove(path);
  }
}
// A bulk add into an empty fast index encodes in chunks; its fast scan equals a full-mode index's fast cache, built
// over every row at once, and a row failing in a later chunk leaves the index empty.
void chunked_add_test(uint32_t bits_per_dimension) {
  constexpr uint32_t dimensions=16,centroids=4;
  constexpr size_t training_rows=512,rows=2*65536+100,failing_row=70000,queries=8,k=10;
  std::mt19937 random(23);std::normal_distribution<float> normal;
  std::vector<float> training(training_rows*dimensions),data(rows*dimensions),probes(queries*dimensions);
  for (auto& value:training) value=normal(random);
  for (auto& value:data) value=normal(random);
  for (auto& value:probes) value=normal(random);
  const auto codec=tq::TieredQ::train({dimensions,bits_per_dimension*dimensions,centroids,false},training.data(),training_rows);
  tq::TieredQIndex chunked(codec),bulk(codec,tq::TieredQIndexMode::full);
  chunked.add(data.data(),rows);
  bulk.add(data.data(),rows); bulk.build_fast_scan();
  auto hits=[&](const tq::TieredQIndex& index) {
    std::pair<std::vector<float>,std::vector<uint32_t>> out{std::vector<float>(queries*k),std::vector<uint32_t>(queries*k)};
    index.search_batch_fast(probes.data(),queries,k,out.first.data(),out.second.data());
    return out;
  };
  const auto expected=hits(bulk);
  require(chunked.size()==rows && hits(chunked)==expected,"chunked add differs from a bulk fast cache");
  auto invalid=data;
  invalid[failing_row*dimensions]=std::nanf("");
  tq::TieredQIndex failed(codec);
  bool rejected=false;
  try { failed.add(invalid.data(),rows); } catch (const std::exception&) { rejected=true; }
  require(rejected && failed.size()==0 && failed.storage_bytes()==0,"failed chunked add leaves rows behind");
  failed.add(data.data(),rows);
  require(hits(failed)==expected,"index does not recover after a failed chunked add");
}
} // namespace
void tieredq_fast_mode_tests() {
  ownership_test(16);ownership_test(64);streamed_load_test();
  for (uint32_t bits:{1u,4u,6u}) chunked_add_test(bits);
}
