// Index ownership, append transactions and exhaustive exact/fast query execution.
#include "tieredq/tieredq.h"
#include "tieredq_io.h"
#include "tieredq_layout.h"
#include "tieredq_search.h"
#include "tieredq_search_fast.h"
#include "tieredq_scan_int8.h"
#include "tieredq_fast_storage.h"
#include "tieredq_fast_dispatch.h"
#include "tieredq_scan_fast_scalar.h"
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define DV_TIEREDQ_X86 1
#include "tieredq_scan_int8_avx512.h"
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#define DV_TIEREDQ_NEON 1
#include "tieredq_scan_int8_neon.h"
#endif
namespace dv {
using tieredq_detail::Layout;
using tieredq_detail::bad;
using tieredq_detail::product;
using tieredq_detail::TopK;
using tieredq_detail::better_hit;
using tieredq_detail::selection_block;
using tieredq_detail::prepare_search_queries;
using tieredq_detail::save_file;
using tieredq_detail::load_file;
std::shared_ptr<tieredq_detail::FastScanCache> TieredQIndex::make_fast_storage(
    const uint8_t* rows,size_t count,bool lookup,int level_max) const {
  if (lookup) {
    auto projection=tieredq_detail::build_fast_query_projection(codec_.config().dimensions,
        codec_.basis_,codec_.sigma_,codec_.mean_,codec_.centroids_,codec_.centroid_half_norms_);
    return tieredq_detail::build_fast_lookup_cache(codec_.levels_,codec_.tiers_,std::move(projection),
        count,codec_.bytes_per_vector(),codec_.code_bytes(),codec_.config().num_centroids,rows);
  }
  return tieredq_detail::build_fast_scan_cache(codec_.levels_,codec_.tiers_,codec_.coded_dimensions(),
      count,codec_.bytes_per_vector(),codec_.code_bytes(),codec_.config().num_centroids,level_max,rows);
}
std::shared_ptr<tieredq_detail::FastScanCache> TieredQIndex::make_fast_storage(const uint8_t* rows,size_t count) const {
#ifdef DV_TIEREDQ_NEON
  return make_fast_storage(rows,count,false,tieredq_detail::arm_fast_level_max);
#else
  return make_fast_storage(rows,count,tieredq_detail::fast_lookup_supported(codec_.tiers_),tieredq_detail::x86_fast_level_max);
#endif
}
bool TieredQIndex::native_fast_scan_available() {
  return tieredq_detail::native_fast_scan_available();
}
bool TieredQIndex::uses_scalar_fast_scan() const {
  return fast_ && tieredq_detail::fast_scan_backend(*fast_)==tieredq_detail::FastScanBackend::scalar;
}
size_t TieredQIndex::payload_bytes() const {
  if (!count_) return 0;
  if (fast_mode_) return count_*bytes_per_vector();
  return count_*codec_.bytes_per_vector();
}
size_t TieredQIndex::storage_bytes() const {
  return payload_.capacity()+(fast_ ? tieredq_detail::fast_storage_bytes(*fast_) : 0);
}
size_t TieredQIndex::bytes_per_vector() const {
  if (!fast_mode_) return codec_.bytes_per_vector();
  if (fast_) return tieredq_detail::fast_vector_bytes(*fast_);
  return tieredq_detail::fast_vector_bytes(*make_fast_storage(nullptr,0));
}
void TieredQIndex::add(const float* rows,size_t count) {
  if (count>UINT32_MAX-size()) bad("index exceeds uint32 document ids");
  auto encoded=codec_.encode(rows,count);
  if (fast_mode_) {
    if (!count) return;
    if (!fast_) fast_=make_fast_storage(encoded.data(),count);
    else {
      auto storage=fast_.use_count()==1 ? fast_ : tieredq_detail::copy_fast_storage(*fast_);
      tieredq_detail::append_fast_rows(*storage,codec_.levels_,codec_.tiers_,codec_.bytes_per_vector(),
          codec_.code_bytes(),codec_.config().num_centroids,encoded.data(),count);
      fast_=std::move(storage);
    }
    count_+=count;
    return;
  }
  const size_t stride=codec_.bytes_per_vector(), total=size()+count;
  payload_.resize(product(total/16+(total%16!=0),product(16,stride)),0);
  for (size_t i=0;i<count;++i) {
    const size_t id=size()+i, block=(id/16)*16*stride, lane=id%16;
    for (size_t byte=0;byte<stride;++byte) payload_[block+byte*16+lane]=encoded[i*stride+byte];
  }
  count_=total;
  if (count) fast_.reset();
}
std::vector<TieredQHit> TieredQIndex::search(const float* vector,size_t k,TieredQKernel kernel) const {
  if (!k || !size()) return {};
  if (fast_mode_) {
    if (kernel!=TieredQKernel::automatic) bad("fast-only search does not support exact kernel overrides");
    return search_fast(vector,k);
  }
  auto q=codec_.query(vector); k=std::min(k,size());
  TopK top(k);
  std::array<float,selection_block> scores;
  const size_t stride=codec_.bytes_per_vector();
  for (size_t begin=0;begin<size();begin+=scores.size()) {
    const size_t n=std::min(scores.size(),size()-begin);
    codec_.score_planes(payload_.data()+begin*stride,n,q,scores.data(),kernel,false);
    top.add(scores.data(),n,begin);
  }
  return top.finish();
}
TieredQIndex::~TieredQIndex() = default;
void TieredQIndex::build_fast_scan() {
  if (fast_mode_) return;
  fast_.reset();
#if defined(DV_TIEREDQ_NEON) || defined(DV_TIEREDQ_X86)
  if (!count_ || !tieredq_detail::fast_scan_available()) return;
  const size_t stride=codec_.bytes_per_vector();
  std::vector<uint8_t> rows(product(count_,stride));
  codec_.unpack(payload_.data(),count_,rows.data());
  fast_=make_fast_storage(rows.data(),count_);
#endif
}
namespace {
template<class Finish>
void fast_search(const tieredq_detail::FastScanCache& cache,const TieredQ& codec,
    const float* queries,size_t rows,size_t k,Finish finish) {
  [[maybe_unused]] const auto backend=tieredq_detail::fast_scan_backend(cache);
#ifdef DV_TIEREDQ_NEON
  if (backend==tieredq_detail::FastScanBackend::neon) {
    auto prepare=[&](const float* vector) { return tieredq_detail::fast_scan_query(cache,codec.query(vector)); };
    auto scan=[&](const auto* prepared,size_t count,size_t first,size_t documents,float* scores,size_t stride) {
      tieredq_detail::scan_fast_neon_tile(cache,prepared,count,first,documents,scores,stride);
    };
    auto prepared=prepare_search_queries(queries,rows,codec.config().dimensions,prepare);
    tieredq_detail::search_fast_tiles(prepared,cache.count,k,tieredq_detail::fast_scan_query_tile_neon,scan,finish);
    return;
  }
#endif
#ifdef DV_TIEREDQ_X86
  if (backend==tieredq_detail::FastScanBackend::avx512) {
    auto prepare=[&](const float* vector) { return tieredq_detail::fast_scan_query_x86(cache,codec,vector); };
    auto scan=[&](const auto* prepared,size_t count,size_t first,size_t documents,float* scores,size_t stride) {
      tieredq_detail::scan_fast_avx512_tile(cache,prepared,count,first,documents,scores,stride);
    };
    const size_t query_tile=cache.lookup ? 1 : tieredq_detail::fast_scan_query_tile_x86;
    auto prepared=prepare_search_queries(queries,rows,codec.config().dimensions,prepare);
    tieredq_detail::search_fast_tiles(prepared,cache.count,k,query_tile,scan,finish);
    return;
  }
#endif
  auto prepared=prepare_search_queries(queries,rows,codec.config().dimensions,
      [&](const float* vector) { return tieredq_detail::fast_scan_query_scalar(cache,codec,vector); });
  auto scan=[&](const auto* prepared,size_t count,size_t first,size_t documents,float* scores,size_t stride) {
    tieredq_detail::scan_fast_scalar_tile(cache,prepared,count,first,documents,scores,stride);
  };
  tieredq_detail::search_fast_tiles(prepared,cache.count,k,1,scan,finish);
}
} // namespace
std::vector<TieredQHit> TieredQIndex::search_fast(const float* vector,size_t k) const {
  if (!k || !size()) return {};
  if (!fast_) return search(vector,k);
  std::vector<TieredQHit> result;
  fast_search(*fast_,codec_,vector,1,std::min(k,size()),
      [&](size_t,std::vector<TieredQHit> hits) { result=std::move(hits); });
  return result;
}
void TieredQIndex::search_batch_fast(const float* queries,size_t rows,size_t k,
    float* scores,uint32_t* ids) const {
  if (!rows || !k || !size()) return;
  if (!queries || !scores || !ids) bad("null search buffer");
  if (!fast_) { search_batch(queries,rows,k,scores,ids); return; }
  const size_t found=std::min(k,size());
  fast_search(*fast_,codec_,queries,rows,found,[&](size_t row,const std::vector<TieredQHit>& hits) {
    for (size_t hit=0;hit<hits.size();++hit) {
      scores[row*found+hit]=hits[hit].score;
      ids[row*found+hit]=hits[hit].id;
    }
  });
}
void TieredQIndex::search_batch(const float* queries,size_t rows,size_t k,
    float* scores,uint32_t* ids) const {
  if (!rows || !k || !size()) return;
  if (!queries || !scores || !ids) bad("null search buffer");
  if (fast_mode_) { search_batch_fast(queries,rows,k,scores,ids); return; }
  const size_t dimensions=codec_.config().dimensions,stride=codec_.bytes_per_vector();
  const size_t found=std::min(k,size());
  const size_t available=std::max<size_t>(1,size_t(omp_get_max_threads()));
  const size_t chunk=std::min<size_t>(4,rows),query_chunks=(rows+chunk-1)/chunk;
  size_t segments=1;
  if (rows>1 && query_chunks<2*available)
    segments=std::min<size_t>(16,(2*available+query_chunks-1)/query_chunks);
  const size_t total_blocks=(size()+selection_block-1)/selection_block;
  size_t segment_blocks=(total_blocks+segments-1)/segments;
  segments=(total_blocks+segment_blocks-1)/segment_blocks;
  const size_t tasks=query_chunks*segments,workers=std::min(tasks,available);
  auto prepared=prepare_search_queries(queries,rows,dimensions,
      [&](const float* vector) { return codec_.query(vector); });
  std::vector<float> partial_scores(tasks*chunk*found);
  std::vector<uint32_t> partial_ids(tasks*chunk*found),partial_counts(tasks*chunk);
  std::exception_ptr failure; std::mutex failure_mutex;
#pragma omp parallel for schedule(dynamic) num_threads(workers) if(tasks>1 && !omp_in_parallel())
  for (size_t task=0;task<tasks;++task) {
    try {
      const size_t first=(task/segments)*chunk,count=std::min(chunk,rows-first);
      const size_t begin=std::min((task%segments)*segment_blocks*selection_block,size());
      const size_t end=std::min(begin+segment_blocks*selection_block,size());
      std::vector<TopK> heaps; heaps.reserve(count);
      for (size_t i=0;i<count;++i) heaps.emplace_back(found);
      std::vector<float> block_scores(count*selection_block);
      for (size_t doc=begin;doc<end;doc+=selection_block) {
        const size_t n=std::min(selection_block,end-doc);
        codec_.score_planes_batch(payload_.data()+doc*stride,n,prepared.data()+first,count,
                                  block_scores.data(),n,TieredQKernel::automatic,false);
        for (size_t i=0;i<count;++i) heaps[i].add(block_scores.data()+i*n,n,doc);
      }
      for (size_t i=0;i<count;++i) {
        const auto hits=heaps[i].finish();
        const size_t offset=(task*chunk+i)*found;
        for (size_t hit=0;hit<hits.size();++hit) {
          partial_scores[offset+hit]=hits[hit].score;
          partial_ids[offset+hit]=hits[hit].id;
        }
        partial_counts[task*chunk+i]=uint32_t(hits.size());
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(failure_mutex);
      if (!failure) failure=std::current_exception();
    }
  }
  if (failure) std::rethrow_exception(failure);
#pragma omp parallel for schedule(static) num_threads(std::min(rows,available)) if(rows>1 && !omp_in_parallel())
  for (size_t row=0;row<rows;++row) {
    const size_t chunk_index=row/chunk,local=row%chunk;
    std::vector<TieredQHit> hits; hits.reserve(segments*found);
    for (size_t segment=0;segment<segments;++segment) {
      const size_t task=chunk_index*segments+segment,offset=(task*chunk+local)*found;
      for (size_t hit=0;hit<partial_counts[task*chunk+local];++hit)
        hits.push_back({partial_scores[offset+hit],partial_ids[offset+hit]});
    }
    std::sort(hits.begin(),hits.end(),better_hit);
    for (size_t hit=0;hit<found;++hit) {
      scores[row*found+hit]=hits[hit].score;
      ids[row*found+hit]=hits[hit].id;
    }
  }
}
} // namespace dv
