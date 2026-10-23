// Fast-only copy-on-write ownership and moved/empty index persistence.
#include "tieredq/tieredq.h"
#include <filesystem>
#include <random>
#include <stdexcept>
#include <unistd.h>
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
  dv::TieredQIndex index(codec,true);
  require(index.fast_mode() && index.storage_bytes()==0,"empty fast-only storage is not empty");
  const auto path=std::filesystem::temp_directory_path()/("tieredq-fast-only-"+std::to_string(getpid())+"-"+std::to_string(bits));
  index.save(path.string());
  auto empty=dv::TieredQIndex::load(path.string());
  require(empty.fast_mode() && empty.size()==0,"empty fast-only mode not persisted");
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
  require(moved.fast_mode() && moved.size()==rows+append_rows && copy.size()==0,"move lost fast-only state");
  moved.save(path.string());
  auto loaded=dv::TieredQIndex::load(path.string());
  loaded.add(data.data(),append_rows);
  require(loaded.fast_mode() && loaded.size()==rows+2*append_rows,"loaded fast-only index cannot append");
  std::filesystem::remove(path);
}
} // namespace
void tieredq_fast_mode_tests() { ownership_test(16);ownership_test(64); }
