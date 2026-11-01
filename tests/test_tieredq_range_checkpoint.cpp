// Range checkpoint rejection safety, score-bit parity, and automatic fast-mode persistence coverage.
#include "tieredq_range_checkpoint.h"
#include "tieredq_train_codebook_data.h"
#include "tieredq_fast_storage.h"
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include "tieredq_range_checkpoint_avx512.h"
#endif
#include <bit>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <unistd.h>

namespace {
void require(bool condition,const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void rejection_test() {
  using namespace dv::tieredq_detail;
  constexpr size_t documents=8213,centroids=2;
  const std::vector<std::pair<uint32_t,uint32_t>> shape{{8,3},{7,5},{6,7},{5,9},{4,1},{3,1}};
  std::vector<dv::TieredQTier> tiers;
  uint32_t dimensions=0,code_bits=0;
  for (const auto& [bits,count]:shape) {
    tiers.push_back({dimensions,count,bits,0,code_bits});
    dimensions+=count;code_bits+=count*bits;
  }
  const size_t code_bytes=(code_bits+7)/8,stride=code_bytes+dv::TieredQ::metadata_bytes;
  const Layout layout{code_bytes};
  std::mt19937 random(41);std::normal_distribution<float> normal;
  std::vector<uint8_t> rows(documents*stride);
  for (size_t doc=0;doc<documents;++doc) {
    auto* row=rows.data()+doc*stride;
    for (const auto& tier:tiers) for (size_t j=0;j<tier.dimensions;++j)
      write_code(row,tier.bit_offset+j*tier.bits,tier.bits,random()%(1u<<tier.bits));
    row[layout.centroid_offset()]=uint8_t(doc%centroids);
    layout.store_half(row,layout.factor_offset(),doc%3==0 ? 0.0f : normal(random));
    layout.store_half(row,layout.correction_offset(),normal(random));
  }
  const auto books=gaussian_codebooks();
  const std::vector<std::vector<float>> levels(books.begin(),books.end());
  auto cache=build_fast_scan_cache(levels,tiers,dimensions,documents,stride,code_bytes,centroids,fast_code_lookup_levels,rows.data());
  require(fast_range_checkpoint_supported(*cache,tiers),"range checkpoint should support wide multi-tier layout");
  auto short_tiers=tiers;short_tiers.resize(fast_range_prefix_tiers);
  require(!fast_range_checkpoint_supported(*cache,short_tiers),"range checkpoint accepted missing suffix");
  dv::TieredQQuery query;
  for (size_t rank=0;rank<dimensions;++rank) query.weights.push_back(normal(random));
  query.centroid_scores={1.0f,-1.0f};query.correction_weights={0.25f,-0.5f};
  const auto bounds=prepare_fast_range_bounds(*cache,tiers,query);
  require(std::isfinite(bounds.high) && std::isfinite(bounds.low) && bounds.high>=bounds.low,
      "range preparation returned invalid endpoints");
  require(fast_vector_bytes(*cache)==cache->code_words()*sizeof(uint32_t)+fast_metadata_bytes,
      "range checkpoint added per-vector metadata");
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512bw")) return;
  std::vector<float> full(documents),actual(documents);
  scan_codes_avx512(*cache,query,0,documents,full.data());
  size_t rejected=0;
  for (float threshold:{-std::numeric_limits<float>::infinity(),0.0f,full[0]}) {
    scan_fast_range_avx512(*cache,query,bounds,tiers[fast_range_prefix_tiers].start,0,documents,actual.data(),threshold);
    for (size_t doc=0;doc<documents;++doc) {
      if (actual[doc]==-std::numeric_limits<float>::infinity()) {
        require(full[doc]<threshold,"range checkpoint rejected a competitive or tied document");
        ++rejected;
      } else require(std::bit_cast<uint32_t>(actual[doc])==std::bit_cast<uint32_t>(full[doc]),
          "range checkpoint changed surviving score bits");
    }
  }
  require(rejected>0,"range checkpoint did not reject any documents");
#endif
}
void automatic_search_test() {
  constexpr uint32_t dimensions=64,centroids=4,bits=256;
  constexpr size_t training_rows=256,documents=8213,append_rows=17,query_count=2;
  std::mt19937 random(43);std::normal_distribution<float> normal;
  std::vector<float> training(training_rows*dimensions),rows((documents+append_rows)*dimensions),queries(query_count*dimensions);
  for (auto* values:{&training,&rows,&queries}) for (size_t i=0;i<values->size();++i)
    (*values)[i]=std::ldexp(normal(random),-int(i%dimensions/8));
  // Full-target training keeps Lloyd-Max books; additive wide tiers would leave the code lookup this test covers.
  dv::TieredQConfig config{dimensions,bits,centroids,false};
  config.target_mode=dv::TieredQIndexMode::full;
  auto codec=dv::TieredQ::train(config,training.data(),training_rows);
  require(codec.tiers().size()>dv::tieredq_detail::fast_range_prefix_tiers,"automatic range test needs a suffix");
  dv::TieredQIndex index(codec,dv::TieredQIndexMode::fast);index.add(rows.data(),documents);
  size_t code_bits=0;
  for (const auto& tier:codec.tiers()) code_bits+=size_t(tier.dimensions)*tier.bits;
  require(index.bytes_per_vector()==(code_bits+31)/32*sizeof(uint32_t)+dv::tieredq_detail::fast_metadata_bytes,
      "automatic range search added stored bits per vector");
  auto compare=[&](const dv::TieredQIndex& candidate) {
    for (size_t k:{size_t(1),size_t(10),candidate.size()+1}) {
      const size_t found=std::min(k,candidate.size());
      std::vector<float> scores(query_count*found);std::vector<uint32_t> ids(query_count*found);
      // Multi-row fast searches retain the unpruned scanner as an independent reference.
      candidate.search_batch_fast(queries.data(),query_count,k,scores.data(),ids.data());
      for (size_t row=0;row<query_count;++row) {
        const auto hits=candidate.search(queries.data()+row*dimensions,k);
        require(hits.size()==found,"automatic range hit count differs");
        for (size_t hit=0;hit<found;++hit)
          require(hits[hit].id==ids[row*found+hit] &&
              std::bit_cast<uint32_t>(hits[hit].score)==std::bit_cast<uint32_t>(scores[row*found+hit]),
              "automatic range search differs from full fast search");
      }
    }
  };
  compare(index);
  const auto path=std::filesystem::temp_directory_path()/("tieredq-range-"+std::to_string(getpid()));
  index.save(path.string());
  const auto file=dv::tieredq_detail::load_file(path.string(),dv::tieredq_detail::fast_index_file_kind);
  dv::tieredq_detail::Reader reader{file.payload};
  require(reader.u32()==dv::tieredq_detail::fast_payload_version,"range checkpoint changed the fast payload version");
  const auto blocks=(documents+dv::tieredq_detail::fast_scan_block_docs-1)/dv::tieredq_detail::fast_scan_block_docs;
  constexpr size_t header_bytes=3*sizeof(uint32_t),vector_headers=4*sizeof(uint64_t);
  require(file.payload.size()==header_bytes+vector_headers+blocks*dv::tieredq_detail::fast_scan_block_docs*index.bytes_per_vector(),
      "range checkpoint changed the serialized per-vector payload");
  auto loaded=dv::TieredQIndex::load(path.string());compare(loaded);
  loaded.add(rows.data()+documents*dimensions,append_rows);compare(loaded);
  loaded.save(path.string());auto reloaded=dv::TieredQIndex::load(path.string());compare(reloaded);
  std::filesystem::remove(path);
}
} // namespace
void tieredq_range_checkpoint_tests() { rejection_test();automatic_search_test(); }
