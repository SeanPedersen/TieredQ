// Streaming fast search crosses document/query tiles without changing scores, ties or ownership.
#include "tieredq/tieredq.h"
#include "tieredq_scan_fast_scalar.h"
#include "tieredq_fast_storage.h"
#include "tieredq_train_codebook_data.h"
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include "tieredq_scan_codes_x86.h"
#endif
#if defined(__aarch64__)
#include "tieredq_scan_codes_neon.h"
#endif
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace {
void require(bool condition,const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void fast_cache_accounting_test() {
  using namespace dv::tieredq_detail;
  const int8_t little_endian_word[]{0x12,0x34,0x56,0x78};
  require(load_code_word(little_endian_word)==0x78563412u,"packed words must be little-endian");
  FastScanCache cache;
  const size_t empty=fast_storage_bytes(cache);
  cache.code_offset.reserve(23);cache.tier_of.reserve(29);cache.level_scale.reserve(7);
  require(fast_storage_bytes(cache)-empty==cache.code_offset.capacity()*sizeof(uint32_t)+
      cache.tier_of.capacity()+cache.level_scale.capacity()*sizeof(float),"fast cache geometry is unaccounted");
  cache.lookup=std::make_shared<FastScanLookupCache>();
  const size_t empty_lookup=fast_storage_bytes(cache);
  auto& lookup=*cache.lookup;
  lookup.projection.correction.reserve(101);lookup.groups.reserve(13);
  lookup.levels.resize(2);lookup.levels[1].reserve(17);
  require(fast_storage_bytes(cache)-empty_lookup==lookup.projection.correction.capacity()*sizeof(float)+
      lookup.groups.capacity()*sizeof(FastLookupGroup)+lookup.levels.capacity()*sizeof(std::vector<float>)+
      lookup.levels[1].capacity()*sizeof(float),"fast lookup shared state is unaccounted");
}
// Decaying coordinate scales mix 5-8-bit tiers with narrow ones that fit a single register-table pair;
// centroid counts above one register pair exercise masked centroid tables.
void high_bit_reference_test(uint32_t bits,float decay=0,uint32_t centroids=4) {
  using namespace dv::tieredq_detail;
  constexpr size_t dimensions=13,documents=4109,query_count=11;
  const size_t training_rows=std::max<size_t>(128,2*centroids);
  std::mt19937 random(123);std::normal_distribution<float> normal;
  std::vector<float> training(training_rows*dimensions),rows(documents*dimensions),queries(query_count*dimensions);
  for (auto* values:{&training,&rows,&queries})
    for (size_t i=0;i<values->size();++i) (*values)[i]=normal(random)*std::exp(-decay*float(i%dimensions));
  // Full-target training keeps Lloyd-Max books, the exact FP32 code-lookup path this test pins.
  dv::TieredQConfig config{dimensions,uint32_t(dimensions*bits),centroids,true};
  config.target_mode=dv::TieredQIndexMode::full;
  auto codec=dv::TieredQ::train(config,training.data(),training_rows);
  const auto encoded=codec.encode(rows.data(),documents);
  auto code_patterns=encoded;
  for (size_t doc=0;doc<documents;++doc) {
    auto* payload=code_patterns.data()+doc*codec.bytes_per_vector();
    std::fill_n(payload,codec.code_bytes(),0);
    for (const auto& tier:codec.tiers()) for (size_t j=0;j<tier.dimensions;++j)
      write_code(payload,tier.bit_offset+j*tier.bits,tier.bits,(doc+j)%(1u<<tier.bits));
  }
  const auto books=gaussian_codebooks();
  const std::vector<std::vector<float>> levels(books.begin(),books.end());
  auto cache=build_fast_scan_cache(levels,codec.tiers(),codec.coded_dimensions(),documents,
      codec.bytes_per_vector(),codec.code_bytes(),centroids,fast_code_lookup_levels,code_patterns.data());
  dv::TieredQIndex index(codec,dv::TieredQIndexMode::fast);index.add(rows.data(),documents);
  const auto batched=codec.query_weights(queries.data(),query_count);
  for (size_t row=0;row<query_count;++row) {
    const auto single=codec.query_weights(queries.data()+row*dimensions);
    require(batched[row].weights==single.weights && batched[row].centroid_scores==single.centroid_scores &&
        batched[row].correction_weights==single.correction_weights,"batched query weights differ from single queries");
  }
  // One call spans full and partial register tiles of every SIMD backend.
  std::vector<dv::TieredQQuery> tile_queries;
  std::vector<float> tile_expected(query_count*documents),tile_actual(query_count*documents);
  for (size_t row=0;row<query_count;++row) {
    tile_queries.push_back(codec.query(queries.data()+row*dimensions));
    codec.score(code_patterns.data(),documents,tile_queries.back(),tile_expected.data()+row*documents,dv::TieredQKernel::scalar);
  }
  [[maybe_unused]] auto tile_query=[&](size_t row)->const dv::TieredQQuery& {return tile_queries[row];};
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  if (__builtin_cpu_supports("avx2")) {
    scan_codes_avx2_tile(*cache,query_count,0,documents,tile_actual.data(),documents,tile_query);
    require(tile_actual==tile_expected,"high-bit AVX2 query tiles differ from scalar codec");
  }
  if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")) {
    scan_codes_avx512_tile(*cache,query_count,0,documents,tile_actual.data(),documents,tile_query);
    require(tile_actual==tile_expected,"high-bit AVX512 query tiles differ from scalar codec");
  }
#endif
#if defined(__aarch64__)
  scan_codes_neon_tile(*cache,query_count,0,documents,tile_actual.data(),documents,tile_query);
  require(tile_actual==tile_expected,"high-bit NEON query tiles differ from scalar codec");
#endif
  for (size_t row=0;row<query_count;++row) {
    const auto q=codec.query(queries.data()+row*dimensions);
    std::vector<float> expected(documents),actual(documents);
    codec.score(code_patterns.data(),documents,q,expected.data(),dv::TieredQKernel::scalar);
    const auto prepared=fast_scan_query_scalar(*cache,codec,queries.data()+row*dimensions);
    scan_fast_scalar_tile(*cache,&prepared,1,0,documents,actual.data(),documents);
    require(actual==expected,"high-bit portable scan lost FP32 codebook precision");
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (__builtin_cpu_supports("avx2")) {
      scan_codes_avx2(*cache,q,0,documents,actual.data());
      require(actual==expected,"high-bit AVX2 scan differs from scalar codec");
    }
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")) {
      scan_codes_avx512(*cache,q,0,documents,actual.data());
      require(actual==expected,"high-bit AVX512 scan differs from scalar codec");
    }
#endif
#if defined(__aarch64__)
    scan_codes_neon(*cache,q,0,documents,actual.data());
    require(actual==expected,"high-bit NEON scan differs from scalar codec");
#endif
    std::vector<uint32_t> order(documents);
    codec.score(encoded.data(),documents,q,expected.data(),dv::TieredQKernel::scalar);
    std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](auto a,auto b){return expected[a]>expected[b] || (expected[a]==expected[b] && a<b);});
    const auto hits=index.search_fast(queries.data()+row*dimensions,10);
    for (size_t hit=0;hit<hits.size();++hit)
      require(hits[hit].id==order[hit] && hits[hit].score==expected[order[hit]],"high-bit native top-k differs from scalar codec");
  }
}
// Hand-built tiers of every width at odd bit offsets exercise each register-table period and phase, straddling
// codes and a partial block against an independent reference that reads the original rows.
void packed_code_layout_test() {
  using namespace dv::tieredq_detail;
  constexpr size_t documents=1000,centroids=3,query_count=11;
  const std::vector<std::pair<uint32_t,uint32_t>> shape{{1,3},{8,37},{7,72},{6,35},{5,70},{4,39},{3,70},{2,37},{1,70}};
  std::vector<dv::TieredQTier> tiers;
  uint32_t dimensions=0,code_bits=0;
  for (const auto& [bits,count]:shape) {
    tiers.push_back({dimensions,count,bits,0,code_bits});
    dimensions+=count;code_bits+=count*bits;
  }
  const size_t code_bytes=(code_bits+7)/8,stride=code_bytes+dv::TieredQ::metadata_bytes;
  const Layout layout{code_bytes};
  std::mt19937 random(31);std::normal_distribution<float> normal;
  std::vector<uint8_t> rows(documents*stride,0);
  for (size_t doc=0;doc<documents;++doc) {
    uint8_t* row=rows.data()+doc*stride;
    for (const auto& tier:tiers) for (size_t j=0;j<tier.dimensions;++j)
      write_code(row,tier.bit_offset+j*tier.bits,tier.bits,random()%(1u<<tier.bits));
    row[layout.centroid_offset()]=uint8_t(random()%centroids);
    layout.store_half(row,layout.factor_offset(),normal(random));
    layout.store_half(row,layout.correction_offset(),normal(random));
  }
  const auto books=gaussian_codebooks();
  const std::vector<std::vector<float>> levels(books.begin(),books.end());
  const auto cache=build_fast_scan_cache(levels,tiers,dimensions,documents,stride,code_bytes,centroids,fast_code_lookup_levels,rows.data());
  require(fast_vector_bytes(*cache)==(code_bits+31)/32*sizeof(uint32_t)+fast_metadata_bytes,"packed code bytes per vector");
  std::vector<dv::TieredQQuery> queries(query_count);
  for (auto& q:queries) {
    for (size_t rank=0;rank<dimensions;++rank) q.weights.push_back(normal(random));
    for (size_t c=0;c<centroids;++c) { q.centroid_scores.push_back(normal(random));q.correction_weights.push_back(normal(random)); }
  }
  std::vector<float> expected(query_count*documents),actual(query_count*documents);
  for (size_t row=0;row<query_count;++row) for (size_t doc=0;doc<documents;++doc) {
    const uint8_t* payload=rows.data()+doc*stride;
    float dot=0;
    for (const auto& tier:tiers) for (size_t j=0;j<tier.dimensions;++j)
      dot+=queries[row].weights[tier.start+j]*books[tier.bits][read_code(payload,tier.bit_offset+j*tier.bits,tier.bits)];
    const size_t cid=payload[layout.centroid_offset()];
    float score=queries[row].centroid_scores[cid];
    score+=queries[row].correction_weights[cid]*layout.correction(payload);
    score+=layout.factor(payload)*dot;
    expected[row*documents+doc]=score;
    actual[row*documents+doc]=fast_code_score(*cache,queries[row],doc);
  }
  require(actual==expected,"packed portable scores differ from row reference");
  [[maybe_unused]] auto query=[&](size_t row)->const dv::TieredQQuery& {return queries[row];};
  [[maybe_unused]] const std::vector<float> single(expected.begin(),expected.begin()+documents);
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  if (__builtin_cpu_supports("avx2")) {
    scan_codes_avx2_tile(*cache,query_count,0,documents,actual.data(),documents,query);
    require(actual==expected,"packed AVX2 tiles differ from row reference");
  }
  if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")) {
    scan_codes_avx512_tile(*cache,query_count,0,documents,actual.data(),documents,query);
    require(actual==expected,"packed AVX512 tiles differ from row reference");
    scan_codes_avx512(*cache,queries[0],0,documents,actual.data());
    require(std::equal(single.begin(),single.end(),actual.begin()),"packed AVX512 single query differs from row reference");
  }
#endif
#if defined(__aarch64__)
  scan_codes_neon_tile(*cache,query_count,0,documents,actual.data(),documents,query);
  require(actual==expected,"packed NEON tiles differ from row reference");
  scan_codes_neon(*cache,queries[0],0,documents,actual.data());
  require(std::equal(single.begin(),single.end(),actual.begin()),"packed NEON single query differs from row reference");
#endif
}
std::vector<uint8_t> file_bytes(const std::filesystem::path& path) {
  std::ifstream in(path,std::ios::binary);
  return {std::istreambuf_iterator<char>(in),{}};
}
// Legacy byte-per-coordinate payloads load into the packed layout and resave canonically; invalid legacy codes fail.
void legacy_byte_payload_test() {
  using namespace dv::tieredq_detail;
  constexpr size_t dimensions=13,documents=37,training_rows=128,centroids=4;
  std::mt19937 random(7);std::normal_distribution<float> normal;
  std::vector<float> training(training_rows*dimensions),rows(documents*dimensions);
  for (auto* values:{&training,&rows})
    for (size_t i=0;i<values->size();++i) (*values)[i]=normal(random)*std::exp(-0.5f*float(i%dimensions));
  // Legacy payloads belong to Lloyd-Max models, which keep the exact code lookup.
  dv::TieredQConfig config{dimensions,uint32_t(dimensions*5),centroids,true};
  config.target_mode=dv::TieredQIndexMode::full;
  auto codec=dv::TieredQ::train(config,training.data(),training_rows);
  dv::TieredQIndex index(codec,dv::TieredQIndexMode::fast);index.add(rows.data(),documents);
  const auto directory=std::filesystem::temp_directory_path();
  const auto packed_path=directory/"tieredq_packed.tq",legacy_path=directory/"tieredq_legacy.tq",resaved_path=directory/"tieredq_resaved.tq";
  index.save(packed_path.string());
  const auto books=gaussian_codebooks();
  const std::vector<std::vector<float>> levels(books.begin(),books.end());
  const auto encoded=codec.encode(rows.data(),documents);
  const auto cache=build_fast_scan_cache(levels,codec.tiers(),codec.coded_dimensions(),documents,
      codec.bytes_per_vector(),codec.code_bytes(),centroids,fast_code_lookup_levels,encoded.data());
  std::vector<uint8_t> bytes(cache->blocks*cache->groups*fast_scan_group_bytes,0);
  for (size_t doc=0;doc<documents;++doc) for (size_t rank=0;rank<cache->dimensions;++rank)
    bytes[fast_level_position(cache->groups,doc,rank)]=uint8_t(fast_code(*cache,doc,rank));
  bytes[fast_level_position(cache->groups,documents,0)]=1; // valid legacy code in a padding document
  auto legacy=[&](const std::vector<uint8_t>& codes) {
    Writer writer;writer.u32(fast_payload_version);writer.u32(fast_byte_code_kind);writer.u32(fast_code_lookup_levels);
    writer.halves(cache->factor.owned());writer.halves(cache->correction.owned());write_fast_bytes(writer,cache->centroid);write_fast_bytes(writer,codes);
    save_file(legacy_path.string(),codec.serialize(),writer.data,documents,fast_index_file_kind);
  };
  legacy(bytes);
  const auto loaded=dv::TieredQIndex::load(legacy_path.string());
  for (size_t doc=0;doc<documents;++doc) {
    const auto expected=index.search_fast(rows.data()+doc*dimensions,5),actual=loaded.search_fast(rows.data()+doc*dimensions,5);
    for (size_t hit=0;hit<expected.size();++hit)
      require(actual[hit].id==expected[hit].id && actual[hit].score==expected[hit].score,"legacy payload search differs");
  }
  loaded.save(resaved_path.string());
  require(file_bytes(resaved_path)==file_bytes(packed_path),"legacy payload does not resave canonically");
  const size_t widest=std::max_element(codec.tiers().begin(),codec.tiers().end(),
      [](const auto& a,const auto& b){return a.bits<b.bits;})->bits;
  size_t narrow=0;
  while (codec.tiers()[cache->tier_of[narrow]].bits==widest) ++narrow;
  bytes[fast_level_position(cache->groups,0,narrow)]=uint8_t(1u<<codec.tiers()[cache->tier_of[narrow]].bits);
  legacy(bytes);
  bool rejected=false;
  try { dv::TieredQIndex::load(legacy_path.string()); } catch (const std::exception&) { rejected=true; }
  require(rejected,"legacy code above its tier width accepted");
  for (const auto& path:{packed_path,legacy_path,resaved_path}) std::filesystem::remove(path);
}
// Signed int8-level payloads written before 5/6-bit models moved to packed codes still load, append and resave.
void legacy_level_payload_test() {
  using namespace dv::tieredq_detail;
  constexpr size_t dimensions=13,documents=37,training_rows=128,centroids=4;
  constexpr uint32_t bits=5;
#if defined(__aarch64__)
  constexpr int level_max=arm_fast_level_max;
#else
  constexpr int level_max=x86_fast_level_max;
#endif
  std::mt19937 random(11);std::normal_distribution<float> normal;
  std::vector<float> training(training_rows*dimensions),rows(documents*dimensions);
  for (auto* values:{&training,&rows}) for (auto& value:*values) value=normal(random);
  // Legacy payloads belong to Lloyd-Max models, which keep the exact code lookup.
  dv::TieredQConfig config{dimensions,uint32_t(dimensions*bits),centroids,true};
  config.target_mode=dv::TieredQIndexMode::full;
  auto codec=dv::TieredQ::train(config,training.data(),training_rows);
  const auto widest=std::max_element(codec.tiers().begin(),codec.tiers().end(),
      [](const auto& a,const auto& b){return a.bits<b.bits;})->bits;
  require(widest>lookup_code_bits && widest<fast_scan_code_bits-1,"legacy level model is not mid-width");
  const auto books=gaussian_codebooks();
  const std::vector<std::vector<float>> levels(books.begin(),books.end());
  const auto encoded=codec.encode(rows.data(),documents);
  const auto cache=build_fast_scan_cache(levels,codec.tiers(),codec.coded_dimensions(),documents,
      codec.bytes_per_vector(),codec.code_bytes(),centroids,level_max,encoded.data());
  const auto directory=std::filesystem::temp_directory_path();
  const auto legacy_path=directory/"tieredq_legacy_levels.tq",resaved_path=directory/"tieredq_legacy_levels_resaved.tq";
  save_file(legacy_path.string(),codec.serialize(),serialize_fast_storage(*cache),documents,fast_index_file_kind);
  auto loaded=dv::TieredQIndex::load(legacy_path.string());
  require(loaded.has_fast_scan() && loaded.search_fast(rows.data(),5).size()==5,"legacy level payload does not search");
  auto check_scores=[&](const auto& reference) {
    const auto prepared=fast_scan_query_scalar(reference,codec,rows.data());
    std::vector<float> expected(reference.count);
    scan_fast_scalar_tile(reference,&prepared,1,0,reference.count,expected.data(),reference.count);
    const auto hits=loaded.search_fast(rows.data(),reference.count);
    for (const auto& hit:hits)
      require(hit.score==expected[hit.id],"legacy level payload changed scores");
  };
  check_scores(*cache);
  loaded.save(resaved_path.string());
  require(file_bytes(resaved_path)==file_bytes(legacy_path),"legacy level payload does not resave unchanged");
  loaded.add(rows.data(),1);
  append_fast_rows(*cache,levels,codec.tiers(),codec.bytes_per_vector(),codec.code_bytes(),centroids,encoded.data(),1);
  check_scores(*cache);
  loaded.save(resaved_path.string());
  const auto file=load_file(resaved_path.string(),fast_index_file_kind);
  Reader reader{file.payload};reader.u32();
  require(reader.u32()==fast_levels_kind && file.rows==documents+1,"appended legacy level payload changed kind");
  for (const auto& path:{legacy_path,resaved_path}) std::filesystem::remove(path);
}
void streaming_search_test(uint32_t code_bits,uint32_t max_scan_bits,bool fast_mode=false) {
  constexpr uint32_t dimensions=8,centroids=4;
  constexpr size_t training_rows=128,documents=4109,query_count=17;
  std::mt19937 random(17);
  std::normal_distribution<float> normal;
  std::vector<float> training(training_rows*dimensions),rows(documents*dimensions),queries(query_count*dimensions);
  for (auto& value:training) value=normal(random);
  for (auto& value:rows) value=normal(random);
  for (auto& value:queries) value=normal(random);
  std::fill_n(queries.data(),dimensions,0);
  for (size_t row=1;row<documents;row+=2) std::copy_n(rows.data(),dimensions,rows.data()+row*dimensions);
  auto codec=dv::TieredQ::train({dimensions,code_bits,centroids,false,max_scan_bits},training.data(),training_rows);
  dv::TieredQIndex index(std::move(codec),fast_mode ? dv::TieredQIndexMode::fast : dv::TieredQIndexMode::full);
  index.add(rows.data(),documents);
  index.build_fast_scan();
  if (!index.has_fast_scan()) return;
  for (size_t k:{size_t(1),size_t(10),documents+1}) {
    const size_t found=std::min(k,documents);
    std::vector<float> scores(query_count*found);
    std::vector<uint32_t> ids(query_count*found);
    index.search_batch_fast(queries.data(),query_count,k,scores.data(),ids.data());
    for (size_t query=0;query<query_count;++query) {
      const auto hits=index.search_fast(queries.data()+query*dimensions,k);
      for (size_t hit=0;hit<found;++hit) {
        const size_t offset=query*found+hit;
        require(ids[offset]==hits[hit].id && scores[offset]==hits[hit].score,"streamed batch differs from single");
        if (!query) require(ids[offset]==hit,"zero-query ties do not retain lowest ids");
      }
    }
  }
}
} // namespace
void tieredq_range_checkpoint_tests();
void tieredq_search_tests() {
  tieredq_range_checkpoint_tests();
  fast_cache_accounting_test();
  high_bit_reference_test(5);high_bit_reference_test(6);high_bit_reference_test(7);high_bit_reference_test(8);
  high_bit_reference_test(5,0.5f,100);
  packed_code_layout_test();legacy_byte_payload_test();legacy_level_payload_test();
  streaming_search_test(16,4);
  streaming_search_test(64,8);
  streaming_search_test(16,4,true);
  streaming_search_test(64,8,true);
  streaming_search_test(56,8,true);
}
