// Fast-only copy-on-write ownership and moved/empty index persistence.
#include "tieredq/tieredq.h"
#include <filesystem>
#include <random>
#include <stdexcept>
#include <unistd.h>
#include <type_traits>
static_assert(!std::is_constructible_v<dv::TieredQIndex,dv::TieredQ,bool>);
namespace {
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
void ownership_test(uint32_t bits) {
  constexpr uint32_t dimensions=8,centroids=4;
  constexpr size_t training_rows=128,rows=257,append_rows=17,k=10;
  std::mt19937 random(19);std::normal_distribution<float> normal;
  std::vector<float> training(training_rows*dimensions),data((rows+append_rows)*dimensions);
  for (auto& value:training) value=normal(random);
  for (auto& value:data) value=normal(random);
  auto codec=dv::TieredQ::train({dimensions,bits,centroids,false},training.data(),training_rows);
  dv::TieredQIndex index(codec);
  try {
    dv::TieredQIndex invalid(codec,static_cast<dv::TieredQIndexMode>(255));
    throw std::runtime_error("invalid mode accepted");
  } catch (const std::invalid_argument&) {}
  require(index.mode()==dv::TieredQIndexMode::fast && index.storage_bytes()==0,"empty fast-only storage is not empty");
  const auto path=std::filesystem::temp_directory_path()/("tieredq-fast-only-"+std::to_string(getpid())+"-"+std::to_string(bits));
  index.save(path.string());
  auto empty=dv::TieredQIndex::load(path.string());
  require(empty.mode()==dv::TieredQIndexMode::fast && empty.size()==0,"empty fast-only mode not persisted");
  empty.add(data.data(),rows);
  index.add(data.data(),rows);
  require(index.uses_scalar_fast_scan()!=dv::TieredQIndex::native_fast_scan_available(),"fast backend diagnostic disagrees with native capability");
  auto before=index.search(data.data(),k);
  auto copy=index;
  copy.add(data.data()+rows*dimensions,append_rows);
  require(index.size()==rows && copy.size()==rows+append_rows,"fast-only copy shares row count");
  const auto after=index.search(data.data(),k);
  for (size_t i=0;i<k;++i)
    require(after[i].id==before[i].id && after[i].score==before[i].score,"append changed a sibling fast-only index");
  auto moved=std::move(copy);
  require(moved.mode()==dv::TieredQIndexMode::fast && moved.size()==rows+append_rows && copy.size()==0,"move lost fast-only state");
  moved.save(path.string());
  auto loaded=dv::TieredQIndex::load(path.string());
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
  require(loaded.mode()==dv::TieredQIndexMode::fast && loaded.size()==rows+2*append_rows,"loaded fast-only index cannot append");
  std::filesystem::remove(path);
}
} // namespace
void tieredq_fast_mode_tests() { ownership_test(16);ownership_test(64); }
