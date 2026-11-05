// TieredQ training, persistence, and shared scalar/SIMD scan kernels.
#include "tieredq/tieredq.h"
#include "tieredq_layout.h"
#include "tieredq_basis_panels.h"
#include "tieredq_search.h"
#include "tieredq_io.h"
#include "tieredq_encode.h"
#include "tieredq_encode_float.h"
#include "tieredq_profile.h"
#include "tieredq_scan_int8.h"
#include "tieredq_train_linalg.h"
#include "tieredq_train_kmeans.h"
#include "tieredq_train_stats.h"
#include "tieredq_train_residuals.h"
#include "tieredq_train_rows.h"
#include "tieredq_train_codebook.h"
#include "tieredq_train_codebook_data.h"
#include "tieredq_train_allocation.h"
#include "tieredq_train_fine.h"
#include "tieredq_metric.h"
#include "training_rng.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <cstring>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <mutex>
#include <omp.h>
#include <stdexcept>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define DV_TIEREDQ_X86 1
#endif
#ifdef DV_TIEREDQ_X86
#include "tieredq_scan_avx512.h"
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#include "tieredq_scan_neon.h"
#define DV_TIEREDQ_NEON 1
#endif
#include "tieredq_scan_groups.h"
namespace dv {
namespace {
using tieredq_detail::Layout;
#ifdef DV_TIEREDQ_X86
using tieredq_detail::half16;
#endif
using tieredq_detail::bad;
using tieredq_detail::product;
using tieredq_detail::Writer;
using tieredq_detail::Reader;
using tieredq_detail::save_file;
using tieredq_detail::load_file;
void finite_rows(const float* x, size_t n) {
  if (n && !x) bad("null vector data");
  for (size_t i = 0; i < n; ++i) if (!std::isfinite(x[i])) bad("nonfinite vector data");
}
// ---- Dense float GEMM: C[rows x padded columns] = A[rows x depth] * B (packed panels) ----
// Reference BLAS is far too slow for encoding; a register-tiled kernel keeps one panel in L2.
constexpr size_t PANEL = TieredQPanels::PANEL, BLOCK_ROWS = 64;
using tieredq_detail::float_matrix;
using tieredq_detail::pack_panels;
// Row-major c[m x n] = a[m x k] * b[k x n], parallel over row blocks.
void multiply(const float* a, int m, int k, const float* b, int n, float* c) {
  const auto panels = pack_panels(k, n, [&](size_t i, size_t j) { return b[i * n + j]; });
  const size_t width = panels.padded_columns(), blocks = (size_t(m) + BLOCK_ROWS - 1) / BLOCK_ROWS;
#pragma omp parallel for schedule(dynamic) if(blocks > 1)
  for (size_t blk = 0; blk < blocks; ++blk) {
    const size_t r0 = blk * BLOCK_ROWS, rows = std::min(BLOCK_ROWS, size_t(m) - r0);
    std::vector<float> wide(rows * width);
    float_matrix(a + r0 * k, rows, k, panels, wide.data(), width);
    for (size_t r = 0; r < rows; ++r) std::copy_n(wide.data() + r * width, n, c + (r0 + r) * n);
  }
}
// Squared-distance argmin via |c|^2 / 2 - <x, c>; ties keep the lowest centroid id.
uint32_t argmin_centroid(const float* dots, const float* half_norms, size_t k) {
  uint32_t id = 0; float best = INFINITY;
  for (size_t c = 0; c < k; ++c) { const float distance = half_norms[c] - dots[c]; if (distance < best) { best = distance; id = uint32_t(c); } }
  return id;
}
std::vector<double> exact_half_norms(const std::vector<float>& centroids,size_t k,size_t d) {
  std::vector<double> out(k);
  for (size_t c=0;c<k;++c) {
    double sum=0; for (size_t j=0;j<d;++j) sum+=double(centroids[c*d+j])*centroids[c*d+j];
    out[c]=sum/2;
  }
  return out;
}
std::vector<float> half_norms(const std::vector<float>& centroids,size_t k,size_t d) {
  const auto exact=exact_half_norms(centroids,k,d);
  return {exact.begin(),exact.end()};
}
void nearest_centroids(const tieredq_detail::TrainingRows& x, size_t rows, size_t d, const std::vector<float>& centroids,
                       size_t k, uint32_t* ids) {
  const auto panels = pack_panels(d, k, [&](size_t i, size_t c) { return centroids[c * d + i]; });
  const auto half = half_norms(centroids, k, d);
  const size_t width = panels.padded_columns(), blocks = (rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
#pragma omp parallel for schedule(dynamic) if(blocks > 1)
  for (size_t blk = 0; blk < blocks; ++blk) {
    const size_t r0 = blk * BLOCK_ROWS, n = std::min(BLOCK_ROWS, rows - r0);
    std::vector<float> dots(n * width), buffer;
    float_matrix(x.block(r0, n, buffer), n, d, panels, dots.data(), width);
    for (size_t r = 0; r < n; ++r) ids[r0 + r] = argmin_centroid(dots.data() + r * width, half.data(), k);
  }
}
// Rows per covariance chunk: a d x chunk f64 block (3 MiB at d = 1536) replaces a d x rows copy of the sample.
constexpr size_t COVARIANCE_CHUNK_ROWS = 256;
// Symmetric covariance in column-major LAPACK layout, accumulated over row chunks.
// fill(r0, n, out) writes centered rows [r0, r0+n) transposed: out[j*n + i] for dimension j.
template<class Fill>
std::vector<double> chunked_covariance(int d, size_t rows, Fill fill) {
  std::vector<double> cov(product(d, d), 0), chunk(product(d, std::min(rows, COVARIANCE_CHUNK_ROWS)));
  for (size_t r0 = 0; r0 < rows; r0 += COVARIANCE_CHUNK_ROWS) {
    const size_t n = std::min(COVARIANCE_CHUNK_ROWS, rows - r0);
    fill(r0, n, chunk.data());
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < d; ++i) for (int j = 0; j <= i; ++j) {
      const double* a = chunk.data() + size_t(i) * n; const double* b = chunk.data() + size_t(j) * n; double s = 0;
#pragma omp simd reduction(+:s)
      for (size_t r = 0; r < n; ++r) s += a[r] * b[r];
      cov[size_t(j) * d + i] += s;
    }
  }
  const double alpha = 1.0 / double(rows - 1);
  for (int i = 0; i < d; ++i) for (int j = 0; j <= i; ++j) cov[size_t(i) * d + j] = cov[size_t(j) * d + i] *= alpha;
  return cov;
}
std::vector<float> rotation(int n, uint64_t seed, const tieredq_detail::TrainingLinalg& linalg, tieredq_detail::StageProfile* sink=nullptr) {
  tieredq_detail::StageProfile profile("tieredq_rotation_profile",sink);
  tieredq_detail::TrainingRng rng(seed);
  std::vector<double> a(product(n,n)), tau(n), signs(n);
  for (int r = 0; r < n; ++r) for (int c = 0; c < n; ++c) a[size_t(c)*n+r] = rng.normal();
  profile.mark("tier_rotation_random_fill");
  double workspace; int lwork = -1, info;
  linalg.qr(&n,&n,a.data(),&n,tau.data(),&workspace,&lwork,&info);
  if (info) bad("QR workspace failed");
  lwork = int(workspace); std::vector<double> work(lwork);
  linalg.qr(&n,&n,a.data(),&n,tau.data(),work.data(),&lwork,&info);
  if (info) bad("QR failed");
  profile.mark("tier_rotation_factor");
  for (int i=0;i<n;++i) signs[i] = a[size_t(i)*n+i] < 0 ? -1 : 1;
  linalg.generate_q(&n,&n,&n,a.data(),&n,tau.data(),work.data(),&lwork,&info);
  if (info) bad("QR generation failed");
  profile.mark("tier_rotation_generate");
  std::vector<float> out(a.size());
  for (int r=0;r<n;++r) for (int c=0;c<n;++c) out[size_t(r)*n+c] = float(a[size_t(c)*n+r]*signs[c]);
  profile.mark("tier_rotation_store");
  return out;
}
// Fine-tier codes wider than 8 bits (#26) span three bytes, which only the scalar scorer reads.
bool has_three_byte_codes(const std::vector<TieredQTier>& tiers) {
  return std::any_of(tiers.begin(),tiers.end(),[](const TieredQTier& t) { return t.bits>tieredq_detail::widest_code_bits; });
}
#ifdef DV_TIEREDQ_X86
// Decode one metadata field for an entire epilogue batch; plane bytes are interleaved into half words.
template<bool transposed>
__attribute__((target("avx2,f16c"))) void half8(const uint8_t* base,size_t stride,size_t offset,float* out) {
  __m128i packed;
  if constexpr (transposed) {
    const auto lo=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(base+offset*16));
    const auto hi=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(base+(offset+1)*16));
    packed=_mm_unpacklo_epi8(lo,hi);
  } else {
    alignas(16) uint16_t values[8];
    for (size_t i=0;i<8;++i) values[i]=uint16_t(base[i*stride+offset]) | uint16_t(base[i*stride+offset+1])<<8;
    packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(values));
  }
  _mm256_storeu_ps(out,_mm256_cvtph_ps(packed));
}
// Keep accumulators in registers for the entire document batch.
// Metadata makes the gather at the last code byte safe for both layouts.
template<bool transposed>
__attribute__((target("avx2,f16c"))) size_t scan8(const uint8_t* payload,size_t count,size_t stride,
    size_t code_bytes,const std::vector<TieredQTier>& tiers,const TieredQQuery& q,
    const Layout& layout,uint32_t centroids,float* scores) {
  if (stride>size_t(INT32_MAX)/8) return 0;
  alignas(64) int positions[8];
  for (int i=0;i<8;++i) positions[i]=int(i*stride);
  const __m256i offsets=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(positions));
  size_t begin=0;
  for (;count-begin>=8;begin+=8) {
    const uint8_t* base=transposed ? payload+(begin/16)*16*stride+begin%16 : payload+begin*stride;
    __m256 sum=_mm256_setzero_ps();
    for (const auto& t:tiers) for (uint32_t j=0;j<t.dimensions;++j) {
      const uint32_t bit=t.bit_offset+j*t.bits;
      __m256i packed;
      if (transposed) {
        packed=_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(base+(bit/8)*16)));
        if (bit%8+t.bits>8) packed=_mm256_or_si256(packed,_mm256_slli_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(base+(bit/8+1)*16))),8));
      } else packed=_mm256_i32gather_epi32(reinterpret_cast<const int*>(base+bit/8), offsets, 1);
      const __m256i ids=_mm256_and_si256(
          _mm256_srlv_epi32(packed,_mm256_set1_epi32(bit%8)),_mm256_set1_epi32((1u<<t.bits)-1));
      const float* table=q.lut.data()+q.offsets[t.start+j];
      sum=_mm256_add_ps(sum,_mm256_i32gather_ps(table, ids, 4));
    }
    alignas(64) float out[8];
    _mm256_storeu_ps(out,sum);
    float factors[8],corr[8];
    { half8<transposed>(base,stride,layout.factor_offset(),factors); half8<transposed>(base,stride,layout.correction_offset(),corr); }
    for (size_t i=0;i<8;++i) {
      const uint8_t* doc=transposed ? base+i : base+i*stride;
      const uint8_t cid=doc[code_bytes*(transposed ? 16 : 1)];
      if (cid>=centroids) bad("invalid centroid id");
      float score=q.centroid_scores[cid];
      score+=q.correction_weights[cid]*corr[i];
      score+=factors[i]*out[i];
      scores[begin+i]=score;
    }
  }
  return begin;
}
// Keep accumulators in registers for the entire document batch.
// Metadata makes the gather at the last code byte safe for both layouts.
template<bool transposed,int batches=1,bool shared=false>
__attribute__((target("avx512f"))) size_t scan16(const uint8_t* payload,size_t count,size_t stride,
    size_t code_bytes,const std::vector<TieredQTier>& tiers,const TieredQQuery& q,
    const Layout& layout,uint32_t centroids,float* scores,
    const std::vector<std::vector<float>>* levels=nullptr) {
  if (stride>size_t(INT32_MAX)/16) return 0;
  alignas(64) int positions[16];
  for (int i=0;i<16;++i) positions[i]=int(i*stride);
  const __m512i offsets=_mm512_loadu_si512(positions);
  size_t begin=0;
  for (;count-begin>=16*batches;begin+=16*batches) {
    const uint8_t* base=payload+begin*stride;
    __m512 sums[batches];
    for (int batch=0;batch<batches;++batch) sums[batch]=_mm512_setzero_ps();
    for (const auto& t:tiers) {
      __m512 book[4]={_mm512_setzero_ps(),_mm512_setzero_ps(),_mm512_setzero_ps(),_mm512_setzero_ps()};
      if (shared && (t.bits==5 || t.bits==6)) {
        book[0]=_mm512_loadu_ps((*levels)[t.bits].data()); book[1]=_mm512_loadu_ps((*levels)[t.bits].data()+16);
        if (t.bits==6) { book[2]=_mm512_loadu_ps((*levels)[t.bits].data()+32); book[3]=_mm512_loadu_ps((*levels)[t.bits].data()+48); }
      }
      __m512 accum[batches];
      for (int batch=0;batch<batches;++batch) accum[batch]=sums[batch];
      for (uint32_t j=0;j<t.dimensions;++j) {
        const uint32_t bit=t.bit_offset+j*t.bits;
        const __m512i shift=_mm512_set1_epi32(bit%8), mask=_mm512_set1_epi32((1u<<t.bits)-1);
        const float* table=q.lut.data()+q.offsets[t.start+j];
        __m512 weight=_mm512_setzero_ps();
        if (shared && t.bits>4) weight=_mm512_set1_ps(q.weights[t.start+j]);
        __m512 lut=_mm512_setzero_ps();
        if (t.bits<=4) lut=_mm512_maskz_loadu_ps(__mmask16((1u<<(1u<<t.bits))-1),table);
        for (int batch=0;batch<batches;++batch) {
          const uint8_t* docs=base+batch*16*stride;
          __m512i packed;
          if (transposed) {
            packed=_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(docs+(bit/8)*16)));
            if (bit%8+t.bits>8) packed=_mm512_or_si512(packed,_mm512_slli_epi32(
                _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(docs+(bit/8+1)*16))),8));
          } else packed=_mm512_i32gather_epi32(offsets,docs+bit/8,1);
          const __m512i ids=_mm512_and_si512(_mm512_srlv_epi32(packed,shift),mask);
          __m512 values;
          if (shared && t.bits==5) values=_mm512_mul_ps(_mm512_permutex2var_ps(book[0],ids,book[1]),weight);
          else if (shared && t.bits==6) {
            const __m512 lo=_mm512_permutex2var_ps(book[0],ids,book[1]),hi=_mm512_permutex2var_ps(book[2],ids,book[3]);
            values=_mm512_mul_ps(_mm512_mask_blend_ps(_mm512_test_epi32_mask(ids,_mm512_set1_epi32(32)),lo,hi),weight);
          } else if (shared && t.bits>4) {
            // 7-8 bit codebooks stay L1 resident; per-coordinate LUTs of that width spill out of L2.
            values=_mm512_mul_ps(_mm512_i32gather_ps(ids,(*levels)[t.bits].data(),4),weight);
          } else values=t.bits<=4 ? _mm512_permutexvar_ps(ids,lut) : _mm512_i32gather_ps(ids,table,4);
          accum[batch]=_mm512_add_ps(accum[batch],values);
        }
      }
      for (int batch=0;batch<batches;++batch) sums[batch]=accum[batch];
    }
    for (int batch=0;batch<batches;++batch) {
      alignas(64) float out[16];
      _mm512_storeu_ps(out,sums[batch]);
      const uint8_t* docs=base+batch*16*stride;
      float factors[16],corr[16];
      { half16<transposed>(docs,stride,layout.factor_offset(),factors); half16<transposed>(docs,stride,layout.correction_offset(),corr); }
      for (size_t i=0;i<16;++i) {
        const uint8_t* doc=transposed ? docs+i : docs+i*stride;
        const uint8_t cid=doc[code_bytes*(transposed ? 16 : 1)];
        if (cid>=centroids) bad("invalid centroid id");
        float score=q.centroid_scores[cid];
        score+=q.correction_weights[cid]*corr[i];
        score+=factors[i]*out[i];
        scores[begin+batch*16+i]=score;
      }
    }
  }
  return begin;
}
#endif
bool same_scan_geometry(const TieredQQuery& a,const TieredQQuery& b) {
  return a.scan_groups.size()==b.scan_groups.size() &&
         std::equal(a.scan_groups.begin(),a.scan_groups.end(),b.scan_groups.begin(),tieredq_detail::same_group);
}
} // namespace
// Tier rotation for other translation units (TieredQRefinement's suffix-only runs).
std::vector<float> tieredq_detail::seeded_rotation(int n,uint64_t seed) { return rotation(n,seed,tieredq_detail::TrainingLinalg::get()); }
TieredQConfig TieredQConfig::target_bytes(uint32_t d,uint32_t bytes) {
  if (bytes>UINT32_MAX/8) bad("byte budget overflow"); return target_code_bits(d,bytes*8);
}
TieredQConfig TieredQConfig::target_code_bits(uint32_t d,uint32_t bits) {
  TieredQConfig c; c.dimensions=d; c.code_bits=bits; c.validate(); return c;
}
void TieredQConfig::validate() const {
  if (!dimensions || dimensions>uint32_t(INT32_MAX) || !code_bits || uint64_t(code_bits)>uint64_t(dimensions)*8)
    bad("dimensions must be positive and budget must be in [1, 8*D]");
  if (!num_centroids || num_centroids>256) bad("centroid count must be in [1, 256]");
  if (max_bits>tieredq_detail::widest_code_bits) bad("max bits per coordinate must be in [1, 8], or 0 for automatic");
  if (fine_tiers!=TieredQFineTiers::automatic && fine_tiers!=TieredQFineTiers::off && fine_tiers!=TieredQFineTiers::on)
    bad("invalid fine-tier mode");
  if (!max_fine_bits || max_fine_bits>tieredq_detail::widest_fine_code_bits) bad("max fine bits must be in [1, 12]");
  if (target_mode!=TieredQIndexMode::fast && target_mode!=TieredQIndexMode::full) bad("invalid target index mode");
  if (metric!=TieredQMetric::inner_product && metric!=TieredQMetric::cosine && metric!=TieredQMetric::l2) bad("invalid metric");
  if (!std::isfinite(allocation_power) || allocation_power<0) bad("allocation power must be finite and non-negative");
}
TieredQ TieredQ::train(TieredQConfig config,const float* sample,size_t rows,uint64_t seed) {
  config.validate(); if (rows<std::max<size_t>(2,size_t(config.num_centroids)*2) || rows>INT32_MAX) bad("training needs at least 2*K rows and at most INT32_MAX rows");
  if (config.metric!=TieredQMetric::inner_product) {
    // Train an inner-product model on the metric's model space, then attach the transform.
    finite_rows(sample,product(rows,config.dimensions));
    TieredQ lifted; lifted.config_=config;
    const size_t d=config.dimensions;
    if (config.metric==TieredQMetric::l2) {
      double peak=0;
      for (size_t i=0;i<rows;++i) {
        double square=0; for (size_t j=0;j<d;++j) square+=double(sample[i*d+j])*sample[i*d+j];
        peak=std::max(peak,square);
      }
      lifted.lift_scale_=peak>0 ? float(std::sqrt(peak)) : 1.0f;
      const size_t calibration=std::min(rows,tieredq_detail::lift_calibration_rows);
      std::vector<float> y(d+1);
      auto moment=chunked_covariance(int(d+1),calibration,[&](size_t r0,size_t n,double* out) {
        for (size_t i=0;i<n;++i) {
          tieredq_detail::l2_lift_input(sample+(r0+i)*d,d,lifted.lift_scale_,true,y.data());
          for (size_t j=0;j<=d;++j) out[j*n+i]=y[j];
        }
      });
      // chunked_covariance normalizes by rows - 1; the lift uses the second moment E[y y^T].
      for (auto& value:moment) value*=double(calibration-1)/double(calibration);
      lifted.lift_base_map_=tieredq_detail::l2_base_map(moment,d+1,tieredq_detail::TrainingLinalg::get().eigen);
      lifted.prepare_lift();
    }
    auto model_config=config;
    model_config.dimensions=uint32_t(lifted.model_dimensions()); model_config.metric=TieredQMetric::inner_product;
    if (!config.allocation_power)
      model_config.allocation_power=config.metric==TieredQMetric::l2 ? 1.0 : 0.0;
    // Cosine rows are normalized on access; the L2 lift is a matrix product, so its rows are lifted once.
    std::vector<float> model_rows;
    auto model_sample=tieredq_detail::TrainingRows{sample,d}.normalized(rows);
    if (config.metric==TieredQMetric::l2) {
      model_rows.resize(product(rows,lifted.model_dimensions()));
      lifted.lift(sample,rows,false,model_rows.data());
      finite_rows(model_rows.data(),model_rows.size());
      model_sample=tieredq_detail::TrainingRows{model_rows.data(),lifted.model_dimensions()};
    }
    // Additive wide tiers lost up to 0.72 pt R@10 on SIFT1M under L2, so its whitened queries keep Lloyd-Max books (#13).
    auto c=train_model(model_config,model_sample,rows,config.metric!=TieredQMetric::l2,seed);
    c.config_.dimensions=config.dimensions; c.config_.metric=config.metric; c.config_.allocation_power=config.allocation_power;
    c.lift_scale_=lifted.lift_scale_; c.lift_base_map_=std::move(lifted.lift_base_map_);
    c.prepare_lift(); c.validate_model();
    return c;
  }
  finite_rows(sample,product(rows,config.dimensions));
  return train_model(config,tieredq_detail::TrainingRows{sample,config.dimensions},rows,true,seed);
}
TieredQ TieredQ::train_sampled(TieredQConfig config,const float* data,size_t count,const uint32_t* sample_ids,
                               size_t sample_rows,uint64_t seed) {
  config.validate();
  if (sample_rows && (!data || !sample_ids)) bad("null training data");
  for (size_t i=0;i<sample_rows;++i) if (sample_ids[i]>=count) bad("training sample id out of range");
  // Metrics with a matrix lift (L2) and tiny samples train on a gathered copy, which is the same model.
  if (config.metric==TieredQMetric::l2 || sample_rows<std::max<size_t>(2,size_t(config.num_centroids)*2) ||
      sample_rows>INT32_MAX) {
    std::vector<float> gathered(product(sample_rows,config.dimensions));
    for (size_t i=0;i<sample_rows;++i)
      std::copy_n(data+size_t(sample_ids[i])*config.dimensions,config.dimensions,gathered.data()+i*config.dimensions);
    return train(config,gathered.data(),sample_rows,seed);
  }
  const tieredq_detail::TrainingRows rows{data,config.dimensions,sample_ids};
  for (size_t i=0;i<sample_rows;++i) finite_rows(rows.row(i),config.dimensions);
  if (config.metric==TieredQMetric::inner_product) return train_model(config,rows,sample_rows,true,seed);
  auto model_config=config;
  model_config.metric=TieredQMetric::inner_product;
  auto c=train_model(model_config,rows.normalized(sample_rows),sample_rows,config.metric!=TieredQMetric::l2,seed);
  c.config_.metric=config.metric; c.config_.allocation_power=config.allocation_power;
  c.prepare_lift(); c.validate_model();
  return c;
}
// Inner-product training on model-space rows, which the caller has checked to be finite.
TieredQ TieredQ::train_model(TieredQConfig config,const tieredq_detail::TrainingRows& sample,size_t rows,bool additive_wide_tiers,uint64_t seed) {
  const int d=int(config.dimensions), n=int(rows); const unsigned k=config.num_centroids;
  tieredq_detail::StageProfile profile;
  const auto& linalg=tieredq_detail::TrainingLinalg::get();
  TieredQ c; c.config_=config;
  tieredq_detail::TrainingRng rng(seed); auto initial=rng.choice(rows,k);
  c.centroids_.resize(product(k,d));
  std::vector<float> buffer;
  for (unsigned j=0;j<k;++j) std::copy_n(sample.block(initial[j],1,buffer),d,c.centroids_.data()+size_t(j)*d);
  std::vector<uint32_t> ids(rows);
  profile.mark("init");
  constexpr int kmeans_iterations=20;
  const int iterations=k==1 ? 1 : kmeans_iterations;
  for (int it=0;it<iterations;++it) {
    if (k>1) nearest_centroids(sample,rows,d,c.centroids_,k,ids.data());
    profile.mark("kmeans_assign");
    sample.visit([&](auto value) { tieredq_detail::update_centroids_with(value,rows,d,ids.data(),c.centroids_); });
    profile.mark("kmeans_update");
  }
  const bool spectral_stats=!std::getenv("DV_TIEREDQ_REFERENCE_STATS");
  std::vector<double> residual,mean,cov;
  if (k>1) nearest_centroids(sample,rows,d,c.centroids_,k,ids.data());
  profile.mark("kmeans_assign");
  if (spectral_stats && linalg.syrk && linalg.small_covariance(d)) {
    auto centered=sample.visit([&](auto value) {
      return tieredq_detail::centered_residuals_with(value,rows,d,ids.data(),c.centroids_);
    });
    mean=std::move(centered.mean);
    profile.mark("residual");
    cov=linalg.covariance_centered(centered.values,d,n);
  } else if (spectral_stats && linalg.syrk) {
    sample.visit([&](auto value) {
      auto difference=[&](size_t i,size_t j) { return double(value(i,j))-c.centroids_[size_t(ids[i])*d+j]; };
      mean.assign(d,0);
      for (size_t i=0;i<rows;++i) for (int j=0;j<d;++j) mean[j]+=difference(i,size_t(j));
      for (int j=0;j<d;++j) mean[j]/=rows;
      profile.mark("residual");
      cov=linalg.covariance_chunked(d,rows,[&](size_t r0,size_t n,double* out) {
#pragma omp parallel for schedule(static)
        for (size_t i=0;i<n;++i) for (int j=0;j<d;++j) out[i*d+j]=difference(r0+i,size_t(j))-mean[j];
      });
    });
  } else {
    // Spectral stats only need centered rows; recomputing differences per chunk avoids any rows x d copy.
    const bool keep_residual=!spectral_stats;
    sample.visit([&](auto value) {
      auto difference=[&](size_t i,int j) { return double(value(i,size_t(j)))-c.centroids_[size_t(ids[i])*d+j]; };
      if (keep_residual) residual.resize(product(rows,d));
      mean.assign(d,0);
      for (size_t i=0;i<rows;++i) for (int j=0;j<d;++j) {
        const double v=difference(i,j);
        if (keep_residual) residual[i*d+j]=v;
        mean[j]+=v;
      }
      for (int j=0;j<d;++j) mean[j]/=rows;
      if (linalg.syrk) cov=linalg.covariance(residual,mean,d,n,profile);
      else {
        profile.mark("residual");
        cov=chunked_covariance(d,rows,[&](size_t r0,size_t n,double* out) {
          for (size_t i=0;i<n;++i) for (int j=0;j<d;++j) out[size_t(j)*n+i]=difference(r0+i,j)-mean[j];
        });
      }
    });
  }
  profile.mark("covariance");
  tieredq_detail::dump_training_stage("covariance.f64",cov);
  std::vector<double> eigenvalues(d);
  const char lower='L', vectors='V';
  int lwork=-1,liwork=-1,info,iquery; double query;
  linalg.eigen(&vectors,&lower,&d,cov.data(),&d,eigenvalues.data(),&query,&lwork,&iquery,&liwork,&info);
  if (info) bad("PCA workspace failed"); lwork=int(query); liwork=iquery;
  std::vector<double> work(lwork); std::vector<int> iwork(liwork);
  linalg.eigen(&vectors,&lower,&d,cov.data(),&d,eigenvalues.data(),work.data(),&lwork,iwork.data(),&liwork,&info);
  if (info) bad("PCA failed");
  profile.mark("eigen");
  tieredq_detail::dump_training_stage("eigenvectors.f64",cov);
  tieredq_detail::dump_training_stage("eigenvalues.f64",eigenvalues);
  c.levels_.resize(tieredq_detail::widest_code_bits+1);
  const auto& additive=tieredq_detail::additive_three_codebook();
  tieredq_detail::DistortionTable gaussian{1};
  for (unsigned b=1;b<=tieredq_detail::widest_code_bits;++b) { c.levels_[b]=tieredq_detail::gaussian_codebooks()[b]; gaussian[b]=tieredq_detail::distortion(c.levels_[b]); }
  auto additive_gaussian=gaussian; additive_gaussian[tieredq_detail::split_code_bits]=tieredq_detail::distortion(additive);
  // Automatic max_bits keeps fast IP/cosine models below fine-section widths: RaBitQ's 4-bit recall on OpenAI then took
  // -0.3/+1.2% code bits and ran 1.4-1.7x faster (#13). Budgets above that per rank still get the width they need.
  const bool fast_additive=config.target_mode==TieredQIndexMode::fast && additive_wide_tiers;
  if (!config.max_bits)
    config.max_bits=fast_additive
        ? std::max(tieredq_detail::additive_wide_max_bits,(config.code_bits+uint32_t(d)-1)/uint32_t(d))
        : tieredq_detail::widest_code_bits;
  // Fine tiers (#26) refine additive prefix books, which exist from the automatic cap's minimum width up.
  const bool fine_possible=fast_additive && config.max_bits>=tieredq_detail::additive_wide_max_bits;
  if (config.fine_tiers==TieredQFineTiers::on && !fine_possible)
    bad("fine tiers need fast-target inner-product or cosine training with max_bits of at least 6");
  if (config.fine_tiers==TieredQFineTiers::on && config.max_fine_bits<=config.max_bits)
    bad("fine tiers need max_fine_bits above max_bits");
  const bool fine_tiers=fine_possible && config.fine_tiers!=TieredQFineTiers::off && config.max_fine_bits>config.max_bits;
  tieredq_detail::AllocationModel allocation{std::vector<double>(d),std::vector<uint32_t>(d),std::vector(d,gaussian)};
  for (int j=0;j<d;++j) {
    allocation.variance[j]=eigenvalues[d-1-j];
    allocation.cap[j]=uint32_t(j)<config.wide_dims ? config.max_bits : std::min<uint32_t>(config.max_bits,6);
  }
  auto bits=tieredq_detail::greedy_allocation(allocation,config.code_bits,false);
  // Allocation uses variance^1.5 instead of the MSE-optimal variance by default from 1 bit per dimension up: for
  // queries distributed like the data it is recall-optimal at equal storage (#9, chosen and confirmed on disjoint
  // query halves). SAQ's inner-product error model puts the optimum at power 2 and ScaNN's parallel-error argument
  // between 1 and 2. Below 1 bit per dimension flatter weights code more ranks and win: power 1 at 0.25-0.5 bits gave
  // +0.7 to +1.0 pt held-out IVF candidate recall on OpenAI-1M and MS MARCO-1M, real and document queries (#22).
  // Whitened queries (the L2 lift) weight every rank alike, so MSE is optimal there and L2 training passes 1.
  const bool fast_target=config.target_mode==TieredQIndexMode::fast;
  const double power=config.allocation_power>0 ? config.allocation_power
                                               : tieredq_detail::default_allocation_power(config.code_bits,uint32_t(d));
  auto weighted=allocation;
  for (auto& v:weighted.variance) v=std::pow(std::max(0.0,v),power);
  // Fast caches of models with every tier <= 4 bits use the x86 nibble lookup, which stores 3-bit codes exactly
  // only for the additive codebook (tieredq_lookup_layout.h). The MSE allocation decides that layout and its padded
  // storage limit, so stored bytes do not depend on the weights; the exact allocator then fills that storage.
  // Full mode and wider fast models pack the Lloyd-Max codes densely.
  if (fast_target && tieredq_detail::widest_tier(bits)<=tieredq_detail::lookup_code_bits) {
    allocation.error.assign(d,additive_gaussian);
    const uint32_t limit=tieredq_detail::lookup_storage_limit(tieredq_detail::lookup_fallback(allocation,config.code_bits,bits),config.code_bits);
    auto fast=tieredq_detail::exact_lookup_allocation(weighted.variance,additive_gaussian,limit,config.max_bits);
    if (fast.additive_three) c.levels_[3]=additive;
    bits=std::move(fast.bits);
  } else if (fast_target && additive_wide_tiers) {
    // Wide tiers use additive books so every code scores through nibble tables (#13). Allocation sees their distortion.
    for (auto& table:weighted.error) {
      table[tieredq_detail::split_code_bits]=tieredq_detail::distortion(additive);
      for (unsigned b=tieredq_detail::lookup_code_bits+1;b<=tieredq_detail::widest_code_bits;++b)
        table[b]=tieredq_detail::distortion(tieredq_detail::additive_wide_codebook(b));
    }
    c.levels_[tieredq_detail::split_code_bits]=additive;
    for (unsigned b=tieredq_detail::lookup_code_bits+1;b<=tieredq_detail::widest_code_bits;++b)
      c.levels_[b]=tieredq_detail::additive_wide_codebook(b);
    bits=tieredq_detail::additive_wide_allocation(weighted,config.code_bits);
    if (fine_tiers) {
      auto books=tieredq_detail::fine_codebooks(config.max_bits);
      if (auto fine=tieredq_detail::fine_allocation(weighted,bits,books,config.max_fine_bits,config.wide_dims,config.code_bits,
                                                    config.fine_tiers==TieredQFineTiers::on)) {
        bits=std::move(*fine);
        c.fast_prefix_bits_=config.max_bits;
        c.levels_.resize(tieredq_detail::widest_fine_code_bits+1);
        for (unsigned b=config.max_bits+1;b<=tieredq_detail::widest_fine_code_bits;++b) c.levels_[b]=std::move(books.books[b]);
      }
    }
  } else if (power!=1.0) {
    bits=tieredq_detail::greedy_allocation(weighted,config.code_bits,false);
  }
  profile.mark("bit_allocation");
  // Group equal precisions; retain source PCA ranks explicitly in the unrotated basis.
  std::vector<int> ranks; uint32_t offset=0;
  for (int b=int(tieredq_detail::widest_fine_code_bits);b>=1;--b) {
    const auto start=ranks.size(); for (int j=0;j<d;++j) if (bits[j]==unsigned(b)) ranks.push_back(j);
    if (ranks.size()!=start) {
      const uint32_t count=uint32_t(ranks.size()-start);
      c.tiers_.push_back({uint32_t(start),count,uint32_t(b),0,offset}); offset+=count*b;
    }
  }
  // Fast nibble allocation bounds padded storage, so it may realise fewer or (filling padding) more bits than
  // requested; the model stores what its codes use.
  c.config_.code_bits=offset;
  const int m=int(ranks.size());
  std::vector<float> pca_basis(product(d,m));
  std::vector<std::vector<float>> rotations;
  for (int j=0;j<d;++j) for (int r=0;r<m;++r) pca_basis[size_t(j)*m+r]=float(cov[size_t(d-1-ranks[r])*d+j]);
  c.basis_=pca_basis;
  profile.mark("basis");
  for (const auto& t:c.tiers_) {
    const int width=int(t.dimensions); std::vector<float> q(product(width,width),0);
    if (config.rotate_tiers) q=rotation(width,seed,linalg,&profile); else for (int j=0;j<width;++j) q[size_t(j)*width+j]=1;
    tieredq_detail::dump_training_stage(("rotation_"+std::to_string(t.start)+".f32").c_str(),q);
    rotations.push_back(q);
    profile.reset();
    std::vector<float> in(product(d,width)),out(in.size());
    for (int j=0;j<d;++j) std::copy_n(pca_basis.data()+size_t(j)*m+t.start,width,in.data()+size_t(j)*width);
    multiply(in.data(),d,width,q.data(),width,out.data());
    for (int j=0;j<d;++j) std::copy_n(out.data()+size_t(j)*width,width,c.basis_.data()+size_t(j)*m+t.start);
    profile.mark("tier_rotation_apply");
  }
  // Serving reads the binary16 basis, so the projection stats are fitted to exactly those values.
  c.basis_half_.resize(c.basis_.size());
  for (size_t i=0;i<c.basis_.size();++i) {
    c.basis_half_[i]=tieredq_detail::half_encode(c.basis_[i]);
    c.basis_[i]=tieredq_detail::half_decode(c.basis_half_[i]);
  }
  if (spectral_stats) {
    if (tieredq_detail::needs_empirical_projection_stats(eigenvalues,ranks))
      tieredq_detail::empirical_projection_stats(sample,rows,ids,c.centroids_,c.basis_,size_t(m),c.mean_,c.sigma_);
    else
      tieredq_detail::spectral_projection_stats(eigenvalues,ranks,c.tiers_,rotations,mean,c.basis_,rows,c.mean_,c.sigma_);
  } else {
    // Residual covariance was centered; restore the mean before fitting the projected mean.
    std::vector<float> rfloat(residual.size());
    for (size_t i=0;i<rows;++i) for (int j=0;j<d;++j) rfloat[i*d+j]=float(residual[i*d+j]);
    std::vector<float> projected(product(rows,m)); multiply(rfloat.data(),n,d,c.basis_.data(),m,projected.data());
    tieredq_detail::projection_stats(projected,rows,m,c.mean_,c.sigma_);
  }
  std::vector<float>().swap(c.basis_);
  c.basis_half_=tieredq_detail::basis_panels_from_rows(c.basis_half_,size_t(d),size_t(m));
  profile.mark("projection_stats");
  c.prepare_serving();
  profile.mark("prepare_serving");
  c.validate_model(); return c;
}

size_t TieredQ::model_dimensions() const { return config_.dimensions+(config_.metric==TieredQMetric::l2 ? 1 : 0); }
void TieredQ::prepare_lift() {
  if (config_.metric!=TieredQMetric::l2) return;
  const size_t lifted=model_dimensions();
  lift_query_map_=tieredq_detail::l2_query_map(lift_base_map_,lifted);
  auto panels=[&](const std::vector<float>& map) { return pack_panels(lifted,lifted,[&](size_t k,size_t c) { return map[k*lifted+c]; }); };
  lift_base_panels_=panels(lift_base_map_); lift_query_panels_=panels(lift_query_map_);
}
void TieredQ::lift(const float* vectors,size_t count,bool queries,float* out) const {
  const size_t d=config_.dimensions,lifted=model_dimensions();
  if (config_.metric==TieredQMetric::inner_product) { std::copy_n(vectors,product(count,d),out); return; }
  if (config_.metric==TieredQMetric::cosine) {
    for (size_t i=0;i<count;++i) tieredq_detail::normalize_row(vectors+i*d,d,out+i*d);
    return;
  }
  const auto& panels=queries ? lift_query_panels_ : lift_base_panels_;
  const size_t width=panels.padded_columns();
  std::vector<float> inputs(product(std::min(count,BLOCK_ROWS),lifted)),wide(product(std::min(count,BLOCK_ROWS),width));
  for (size_t r0=0;r0<count;r0+=BLOCK_ROWS) {
    const size_t n=std::min(BLOCK_ROWS,count-r0);
    for (size_t i=0;i<n;++i) tieredq_detail::l2_lift_input(vectors+(r0+i)*d,d,lift_scale_,queries,inputs.data()+i*lifted);
    float_matrix(inputs.data(),n,lifted,panels,wide.data(),width);
    for (size_t i=0;i<n;++i) std::copy_n(wide.data()+i*width,lifted,out+(r0+i)*lifted);
  }
}
void TieredQ::lift_queries(const float* queries,size_t count,float* out) const {
  finite_rows(queries,product(count,config_.dimensions));
  lift(queries,count,true,out);
}
std::pair<double,double> TieredQ::score_map(const float* query) const {
  if (config_.metric!=TieredQMetric::l2) return {1.0,0.0};
  double square=0; for (size_t j=0;j<config_.dimensions;++j) square+=double(query[j])*query[j];
  return {2.0*double(lift_scale_)*lift_scale_,-square};
}
void TieredQ::prepare_serving() {
  prepare_lift();
  centroid_half_norms_=half_norms(centroids_,config_.num_centroids,model_dimensions());
  tieredq_detail::build_scan_plan(tiers_,scan_plan_);
  encoder_=std::make_shared<tieredq_detail::LazyEncoder>();
}
tieredq_detail::EncoderData TieredQ::build_encoder() const {
  const size_t d=model_dimensions(),m=coded_dimensions(),k=config_.num_centroids;
  tieredq_detail::EncoderData encoder;
  encoder.fused_panels=pack_panels(d,m+k,[&](size_t j,size_t r) {
    return r<m ? basis_value(j*m+r) : centroids_[(r-m)*d+j];
  });
  const auto projection=pack_panels(d,m,[&](size_t j,size_t r) { return basis_value(j*m+r); });
  const std::vector<double> centers(centroids_.begin(),centroids_.end());
  encoder.centroid_projections.resize(product(k,projection.padded_columns()));
  tieredq_detail::ordered_matrix(centers.data(),k,projection,encoder.centroid_projections.data());
  encoder.half_norms=exact_half_norms(centroids_,k,d);
  return encoder;
}
// One fused GEMM yields the basis projection [0, m) and centroid dots [m, m+K) per row;
// the residual projection is x*basis - c*basis.
void TieredQ::encode_block(const float* rows,size_t count,uint8_t* out,float* corrections,tieredq_detail::StageProfile* profile) const {
  const Layout layout{code_bytes()};
  const tieredq_detail::EncodingModel model{model_dimensions(),centroids_,mean_,sigma_,levels_,tiers_};
  std::vector<float> lifted;
  if (config_.metric!=TieredQMetric::inner_product) {
    lifted.resize(product(count,model_dimensions())); lift(rows,count,false,lifted.data()); rows=lifted.data();
  }
  const auto encoder=encoder_->get([&] { return build_encoder(); }).view();
  const TieredQKernel kernel=kernel_available(TieredQKernel::neon) ? TieredQKernel::neon : TieredQKernel::automatic;
  tieredq_detail::encode_block(model,encoder,layout,rows,count,out,corrections,kernel,profile);
}
void TieredQ::encode_rows(const float* rows,size_t count,uint8_t* out,float* corrections) const {
  const size_t d=config_.dimensions,stride=bytes_per_vector(),blocks=(count+BLOCK_ROWS-1)/BLOCK_ROWS;
  tieredq_detail::StageProfile profile("tieredq_encode_worker_seconds");
  std::exception_ptr failure;
#pragma omp parallel for schedule(dynamic) if(blocks > 1)
  for (size_t blk=0;blk<blocks;++blk) {
    const size_t r0=blk*BLOCK_ROWS;
    try { encode_block(rows+r0*d,std::min(BLOCK_ROWS,count-r0),out+r0*stride,corrections ? corrections+r0 : nullptr,&profile);
    } catch (...) {
#pragma omp critical(tieredq_encode_failure)
      { if (!failure) failure=std::current_exception(); }
    }
  }
  if (failure) std::rethrow_exception(failure);
}
void TieredQ::encode(const float* rows,size_t count,uint8_t* out,float* corrections) const {
  finite_rows(rows,product(count,config_.dimensions));
  if (count && !out) bad("null encoding buffer");
  encode_rows(rows,count,out,corrections);
}
std::vector<uint8_t> TieredQ::encode(const float* rows,size_t count) const {
  std::vector<uint8_t> out(product(count,bytes_per_vector()));
  encode(rows,count,out.data());
  return out;
}
TieredQQuery TieredQ::query(const float* vector) const {
  finite_rows(vector,config_.dimensions);
  std::vector<float> lifted(model_dimensions());
  lift(vector,1,true,lifted.data());
  return model_query(lifted.data());
}
TieredQQuery TieredQ::model_query(const float* vector) const {
  auto q=model_query_weights(vector);
  q.offsets.resize(coded_dimensions());
  for (const auto& t:tiers_) for (uint32_t j=0;j<t.dimensions;++j) {
    const size_t rank=t.start+j;q.offsets[rank]=q.lut.size();
    for (float level:levels_[t.bits]) q.lut.push_back(q.weights[rank]*level);
  }
  tieredq_detail::build_scan_groups(tiers_,q);
  return q;
}
namespace {
// Queries per projection tile and interleaved centroid reductions; tiles share each basis row and run
// centroid dots across query lanes. Every output keeps its per-query summation order (input dimensions
// ascending), so batching never changes a weight or score.
constexpr size_t query_weight_tile=8,query_centroid_chains=4;
template<size_t Chains>
void centroid_dot_tile(const float* lanes,const float* centroids,size_t d,size_t first,float (&dots)[Chains][query_weight_tile]) {
  for (auto& chain:dots) std::fill(std::begin(chain),std::end(chain),0.0f);
  for (size_t i=0;i<d;++i) {
    const float* lane=lanes+i*query_weight_tile;
    for (size_t chain=0;chain<Chains;++chain) {
      const float value=centroids[(first+chain)*d+i];
      for (size_t row=0;row<query_weight_tile;++row) dots[chain][row]+=lane[row]*value;
    }
  }
}
} // namespace
void TieredQ::project_query_tile(const float* vectors,size_t rows,TieredQQuery* out) const {
  const size_t m=coded_dimensions(),d=model_dimensions(),centroids=config_.num_centroids;
  std::vector<float> transformed(rows*m,0);
  if (!basis_half_.empty())
    tieredq_detail::project_basis_panels({basis_half_.data(),d,m,nullptr,vectors,d,transformed.data(),m},rows);
  else for (size_t i=0;i<d;++i) {
    const float* basis=basis_.data()+i*m;
    for (size_t row=0;row<rows;++row) {
      const float value=vectors[row*d+i];
      float* projected=transformed.data()+row*m;
      for (size_t j=0;j<m;++j) projected[j]+=value*basis[j];
    }
  }
  std::array<float,query_weight_tile> correction{};
  for (size_t row=0;row<rows;++row) {
    const float* projected=transformed.data()+row*m;
    auto& q=out[row];
    q.weights.resize(m);
    for (size_t rank=0;rank<m;++rank) q.weights[rank]=projected[rank]*sigma_[rank];
    // Quantization removes the projected residual mean; restore it in the score.
    for (size_t j=0;j<m;++j) correction[row]+=projected[j]*mean_[j];
  }
  for (size_t row=0;row<rows;++row) {
    out[row].centroid_scores.assign(centroids,0); out[row].correction_weights.assign(centroids,0);
  }
  std::vector<float> lanes(d*query_weight_tile,0); // input dimension major, zero padded query lanes
  for (size_t row=0;row<rows;++row) for (size_t i=0;i<d;++i) lanes[i*query_weight_tile+row]=vectors[row*d+i];
  auto finish=[&](size_t id,const float* dots) {
    const float norm=2*centroid_half_norms_[id];
    for (size_t row=0;row<rows;++row) {
      out[row].centroid_scores[id]=dots[row]+correction[row];
      out[row].correction_weights[id]=norm>0 ? dots[row]/norm : 0;
    }
  };
  size_t id=0;
  for (;id+query_centroid_chains<=centroids;id+=query_centroid_chains) {
    float dots[query_centroid_chains][query_weight_tile];
    centroid_dot_tile(lanes.data(),centroids_.data(),d,id,dots);
    for (size_t chain=0;chain<query_centroid_chains;++chain) finish(id+chain,dots[chain]);
  }
  for (;id<centroids;++id) {
    float dots[1][query_weight_tile];
    centroid_dot_tile(lanes.data(),centroids_.data(),d,id,dots);
    finish(id,dots[0]);
  }
}
TieredQQuery TieredQ::query_weights(const float* vector) const {
  finite_rows(vector,config_.dimensions);
  std::vector<float> lifted(model_dimensions());
  lift(vector,1,true,lifted.data());
  return model_query_weights(lifted.data());
}
std::vector<TieredQQuery> TieredQ::query_weights(const float* vectors,size_t rows) const {
  finite_rows(vectors,product(rows,config_.dimensions));
  if (config_.metric==TieredQMetric::inner_product) return model_query_weights(vectors,rows);
  std::vector<float> lifted(product(rows,model_dimensions()));
  lift(vectors,rows,true,lifted.data());
  return model_query_weights(lifted.data(),rows);
}
TieredQQuery TieredQ::model_query_weights(const float* vector) const {
  finite_rows(vector,model_dimensions());
  TieredQQuery q;
  project_query_tile(vector,1,&q);
  return q;
}
std::vector<TieredQQuery> TieredQ::model_query_weights(const float* vectors,size_t rows) const {
  finite_rows(vectors,product(rows,model_dimensions()));
  std::vector<TieredQQuery> queries(rows);
  const size_t tile=tieredq_detail::query_tile_height(rows,query_weight_tile),tiles=(rows+tile-1)/tile;
  const size_t workers=std::max<size_t>(1,std::min(tiles,size_t(omp_get_max_threads())));
  std::exception_ptr failure; std::mutex failure_mutex;
#pragma omp parallel for schedule(static) num_threads(workers) if(tiles>1 && !omp_in_parallel())
  for (size_t index=0;index<tiles;++index) {
    try {
      const size_t first=index*tile;
      project_query_tile(vectors+first*model_dimensions(),std::min(tile,rows-first),queries.data()+first);
    } catch (...) {
      std::lock_guard<std::mutex> lock(failure_mutex);
      if (!failure) failure=std::current_exception();
    }
  }
  if (failure) std::rethrow_exception(failure);
  return queries;
}
bool TieredQ::kernel_available(TieredQKernel kernel) {
  if (kernel==TieredQKernel::automatic || kernel==TieredQKernel::scalar) return true;
#ifdef DV_TIEREDQ_NEON
  if (kernel==TieredQKernel::neon) return true;
#endif
#ifdef DV_TIEREDQ_X86
  if (kernel==TieredQKernel::avx2) return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
  if (kernel==TieredQKernel::avx512) return __builtin_cpu_supports("avx512f");
#endif
  return false;
}
void TieredQ::score(const uint8_t* payload,size_t count,const TieredQQuery& q,float* scores,TieredQKernel kernel) const {
  const Layout layout{code_bytes()};
  if (count && (!payload || !scores)) bad("null scoring buffer");
  layout.validate_weights(q.correction_weights.size(),config_.num_centroids);
  product(count,bytes_per_vector());
  if (q.offsets.size()!=coded_dimensions() || q.centroid_scores.size()!=config_.num_centroids) bad("query shape mismatch");
  size_t expected=0;
  for (const auto& t:tiers_) for (uint32_t j=0;j<t.dimensions;++j) {
    if (q.offsets[t.start+j]!=expected) bad("query LUT offsets mismatch"); expected+=size_t(1)<<t.bits;
  }
  if (q.lut.size()!=expected) bad("query LUT size mismatch");
  if (!q.scan_groups.empty() && !tieredq_detail::scan_groups_match(tiers_,q))
    bad("query scan group mismatch");
  if (kernel==TieredQKernel::automatic) kernel=kernel_available(TieredQKernel::avx512) ? TieredQKernel::avx512 : kernel_available(TieredQKernel::avx2) ? TieredQKernel::avx2 : TieredQKernel::scalar;
  if (!kernel_available(kernel)) bad("requested scoring kernel is unavailable");
  if (has_three_byte_codes(tiers_)) kernel=TieredQKernel::scalar;
  if (!count) return;
  const size_t stride=bytes_per_vector();
  size_t begin=0;
#ifdef DV_TIEREDQ_X86
  if (kernel==TieredQKernel::avx512) {
    begin=scan16<false,4>(payload,count,stride,code_bytes(),tiers_,q,layout,config_.num_centroids,scores);
    begin+=scan16<false>(payload+begin*stride,count-begin,stride,code_bytes(),tiers_,q,layout,config_.num_centroids,scores+begin);
  }
  else if (kernel==TieredQKernel::avx2) begin=scan8<false>(payload,count,stride,code_bytes(),tiers_,q,layout,config_.num_centroids,scores);
#endif
  for (;begin<count;++begin) {
    const uint8_t* doc=payload+begin*stride; float sum=0;
    for (const auto& t:tiers_) for (uint32_t j=0;j<t.dimensions;++j)
      sum+=q.lut[q.offsets[t.start+j]+tieredq_detail::read_wide_code(doc,t.bit_offset+j*t.bits,t.bits)];
    const uint8_t cid=doc[code_bytes()];
    if (cid>=config_.num_centroids) bad("invalid centroid id");
    float score=q.centroid_scores[cid];
    score+=q.correction_weights[cid]*layout.correction(doc);
    score+=layout.factor(doc)*sum;
    scores[begin]=score;
  }
}
size_t TieredQ::metadata_bytes_per_vector() const { return Layout{code_bytes()}.metadata_bytes(); }
size_t TieredQ::model_bytes() const { return serialize().size(); }
void TieredQ::validate_model() const {
  config_.validate(); const size_t d=model_dimensions(),m=sigma_.size();
  if (config_.metric==TieredQMetric::l2) {
    if (!(lift_scale_>0) || !std::isfinite(lift_scale_) || lift_base_map_.size()!=product(d,d)) bad("invalid L2 lift");
    finite_rows(lift_base_map_.data(),lift_base_map_.size());
    for (size_t i=0;i<d;++i) {
      if (!(lift_base_map_[i*d+i]>0)) bad("invalid L2 lift");
      for (size_t j=i+1;j<d;++j) if (lift_base_map_[i*d+j]!=0) bad("L2 lift is not lower triangular");
    }
  } else if (!lift_base_map_.empty() || lift_scale_!=1) bad("lift on a non-L2 model");
  const size_t tables=(fast_prefix_bits_ ? tieredq_detail::widest_fine_code_bits : tieredq_detail::widest_code_bits)+1;
  if (!m || m>d || tiers_.empty() || tiers_.size()>tieredq_detail::widest_fine_code_bits || levels_.size()!=tables)
    bad("invalid model layout");
  if (fast_prefix_bits_ && (fast_prefix_bits_<tieredq_detail::additive_wide_max_bits || fast_prefix_bits_>tieredq_detail::widest_code_bits ||
      tiers_.front().bits<=fast_prefix_bits_))
    bad("invalid fine-tier prefix");
  if (!basis_.empty() && !basis_half_.empty()) bad("model holds two basis encodings");
  const bool basis_shape=basis_half_.empty() ? basis_.size()==product(d,m) :
      basis_half_.size()==tieredq_detail::basis_panel_size(d,m) && tieredq_detail::basis_panel_padding_zero(basis_half_,d,m);
  if (centroids_.size()!=product(config_.num_centroids,d) || !basis_shape || mean_.size()!=m) bad("invalid model array shape");
  uint32_t rank=0; uint64_t offset=0;
  for (size_t i=0;i<tiers_.size();++i) {
    const auto& t=tiers_[i];
    if (t.start!=rank || !t.dimensions || t.dimensions>m-rank || !t.bits || t.bits>=levels_.size() || t.scale_group!=0 || t.bit_offset!=offset) bad("invalid tier descriptor");
    rank+=t.dimensions; offset+=uint64_t(t.dimensions)*t.bits;
  }
  if (rank!=m || offset!=config_.code_bits) bad("tier budget mismatch");
  if (!levels_[0].empty()) bad("invalid zero-bit table");
  for (unsigned b=1;b<levels_.size();++b) if (levels_[b].size()!=(size_t(1)<<b) || !std::is_sorted(levels_[b].begin(),levels_[b].end())) bad("invalid Lloyd-Max table");
  for (float s:sigma_) if (!(s>0) || !std::isfinite(s)) bad("invalid sigma");
  for (const auto* a:{&centroids_,&basis_,&mean_}) finite_rows(a->data(),a->size());
  for (uint16_t v:basis_half_) if (!std::isfinite(tieredq_detail::half_decode(v))) bad("nonfinite model");
  for (const auto& a:levels_) finite_rows(a.data(),a.size());
}
std::vector<uint8_t> TieredQ::serialize() const {
  validate_model(); Writer w;
  w.u32(config_.dimensions); w.u32(config_.code_bits); w.u32(config_.num_centroids);
  const bool half_basis=!basis_half_.empty();
  const bool metric=config_.metric!=TieredQMetric::inner_product;
  if (metric && !half_basis) bad("metric models need a binary16 basis");
  w.u32(config_.rotate_tiers); w.u32(metric ? metric_model_format : half_basis ? model_format : fp32_basis_model_format);
  w.u32(uint32_t(code_bytes())); w.u32(uint32_t(bytes_per_vector()));
  w.u32(uint32_t(tiers_.size()));
  for (const auto& t:tiers_) { w.u32(t.start); w.u32(t.dimensions); w.u32(t.bits); w.u32(t.scale_group); w.u32(t.bit_offset); }
  w.floats(centroids_);
  if (half_basis) w.halves(tieredq_detail::basis_rows_from_panels(basis_half_,model_dimensions(),coded_dimensions())); else w.floats(basis_);
  w.floats(mean_); w.floats(sigma_);
  w.u32(uint32_t(levels_.size())); for (const auto& l:levels_) w.floats(l);
  // Fine-tier models (#26) alone hold the wider tables, followed by their prefix width.
  if (fast_prefix_bits_) w.u32(fast_prefix_bits_);
  if (metric) {
    w.u32(uint32_t(config_.metric));
    if (config_.metric==TieredQMetric::l2) {
      // Only the lower triangle of the base map is stored.
      const size_t lifted=model_dimensions(); std::vector<float> lower;
      for (size_t i=0;i<lifted;++i) for (size_t j=0;j<=i;++j) lower.push_back(lift_base_map_[i*lifted+j]);
      w.floats({lift_scale_}); w.floats(lower);
    }
  }
  return w.data;
}
TieredQ TieredQ::prefix_view(std::vector<TieredQTier> tiers) const {
  TieredQ view=*this;
  view.config_.code_bits=0;
  for (const auto& t:tiers) view.config_.code_bits+=t.dimensions*t.bits;
  view.tiers_=std::move(tiers);
  view.fast_prefix_bits_=0;
  view.levels_.resize(tieredq_detail::widest_code_bits+1);
  return deserialize(view.serialize());
}
TieredQ TieredQ::deserialize(const std::vector<uint8_t>& data) {
  Reader r{data}; TieredQ c;
  c.config_.dimensions=r.u32(); c.config_.code_bits=r.u32(); c.config_.num_centroids=r.u32();
  const auto rotate=r.u32(); if (rotate>1) bad("invalid rotation flag"); c.config_.rotate_tiers=rotate;
  const auto format=r.u32();
  if (format!=metric_model_format && format!=model_format && format!=fp32_basis_model_format) bad("unsupported codec format");
  c.config_.validate();
  auto cb=r.u32(),pb=r.u32(),nt=r.u32(); if (cb!=c.code_bytes() || nt==0 || nt>tieredq_detail::widest_fine_code_bits) bad("invalid layout header");
  for (uint32_t i=0;i<nt;++i) c.tiers_.push_back({r.u32(),r.u32(),r.u32(),r.u32(),r.u32()});
  c.centroids_=r.floats();
  if (format==fp32_basis_model_format) c.basis_=r.floats(); else c.basis_half_=r.halves();
  c.mean_=r.floats(); c.sigma_=r.floats();
  const uint32_t table_count=r.u32();
  if (table_count!=tieredq_detail::widest_code_bits+1 && table_count!=tieredq_detail::widest_fine_code_bits+1)
    bad("table count mismatch");
  for (uint32_t i=0;i<table_count;++i) c.levels_.push_back(r.floats());
  if (table_count==tieredq_detail::widest_fine_code_bits+1) c.fast_prefix_bits_=r.u32();
  if (format==metric_model_format) {
    const uint32_t metric=r.u32();
    if (metric!=uint32_t(TieredQMetric::cosine) && metric!=uint32_t(TieredQMetric::l2)) bad("invalid metric");
    c.config_.metric=TieredQMetric(metric);
    if (c.config_.metric==TieredQMetric::l2) {
      const auto scale=r.floats(); const auto lower=r.floats();
      const size_t lifted=c.model_dimensions();
      if (scale.size()!=1 || lower.size()!=lifted*(lifted+1)/2) bad("invalid L2 lift");
      c.lift_scale_=scale[0]; c.lift_base_map_.assign(lifted*lifted,0);
      size_t at=0;
      for (size_t i=0;i<lifted;++i) for (size_t j=0;j<=i;++j) c.lift_base_map_[i*lifted+j]=lower[at++];
    }
  }
  if (r.pos!=data.size() || pb!=c.bytes_per_vector()) bad("metadata length mismatch");
  // The metric fixes the model dimensions, so the file's row-major basis becomes panels last.
  if (!c.basis_half_.empty()) {
    if (c.basis_half_.size()!=product(c.model_dimensions(),c.coded_dimensions())) bad("invalid model array shape");
    c.basis_half_=tieredq_detail::basis_panels_from_rows(c.basis_half_,c.model_dimensions(),c.coded_dimensions());
  }
  c.validate_model(); c.prepare_serving(); return c;
}
void TieredQ::save(const std::string& path) const { save_file(path,serialize(),{},0,tieredq_detail::codec_file_kind); }
TieredQ TieredQ::load(const std::string& path) { return deserialize(load_file(path,tieredq_detail::codec_file_kind).model); }
void TieredQ::score_planes(const uint8_t* block,size_t n,const TieredQQuery& q,float* scores,TieredQKernel kernel,bool validate_query) const {
  const Layout layout{code_bytes()};
  if (!n) return;
  if (!block || !scores) bad("null scoring buffer");
  layout.validate_weights(q.correction_weights.size(),config_.num_centroids);
  // Trusted engine callers reuse an unmodified query generated by this model.
  if (validate_query) score(nullptr,0,q,nullptr,TieredQKernel::scalar);
  if (q.weights.size()!=coded_dimensions()) bad("query weight shape mismatch");
  if (kernel==TieredQKernel::automatic) kernel=kernel_available(TieredQKernel::avx512) ? TieredQKernel::avx512 : kernel_available(TieredQKernel::avx2) ? TieredQKernel::avx2 : kernel_available(TieredQKernel::neon) ? TieredQKernel::neon : TieredQKernel::scalar;
  if (!kernel_available(kernel)) bad("requested scoring kernel is unavailable");
  if (has_three_byte_codes(tiers_)) kernel=TieredQKernel::scalar;
  const size_t stride=bytes_per_vector(); size_t done=0;
  std::vector<uint8_t> tail;
#ifdef DV_TIEREDQ_NEON
  if (kernel==TieredQKernel::neon && !q.scan_groups.empty())
    done=tieredq_detail::scan_neon(block,n,stride,q,layout,config_.num_centroids,scores);
#endif
#ifdef DV_TIEREDQ_X86
    if (kernel==TieredQKernel::avx512 && !q.scan_groups.empty() && __builtin_cpu_supports("avx512bw")) {
      done=tieredq_detail::scan_grouped<4>(block,n,stride,code_bytes(),q,layout,config_.num_centroids,levels_,scan_plan_,scores);
      done+=tieredq_detail::scan_grouped<1>(block+done*stride,n-done,stride,code_bytes(),q,layout,config_.num_centroids,levels_,scan_plan_,scores+done);
    } else if (kernel==TieredQKernel::avx512) {
      done=scan16<true,4,true>(block,n,stride,code_bytes(),tiers_,q,layout,config_.num_centroids,scores,&levels_);
      done+=scan16<true,1,true>(block+done*stride,n-done,stride,code_bytes(),tiers_,q,layout,config_.num_centroids,scores+done,&levels_);
    }
    else if (kernel==TieredQKernel::avx2) done=scan8<true>(block,n,stride,code_bytes(),tiers_,q,layout,config_.num_centroids,scores);
#endif
    if (done<n) tail.resize(16*stride);
    while (done<n) {
      size_t rows=std::min<size_t>(16,n-done);
      for (size_t i=0;i<rows;++i) for (size_t byte=0;byte<stride;++byte)
        tail[i*stride+byte]=block[((done+i)/16)*16*stride+byte*16+(done+i)%16];
      score(tail.data(),rows,q,scores+done,TieredQKernel::scalar);
      done+=rows;
    }
}
void TieredQ::score_planes_batch(const uint8_t* block,size_t n,const TieredQQuery* queries,
    size_t query_count,float* scores,size_t score_stride,TieredQKernel kernel,bool validate_query) const {
  const Layout layout{code_bytes()};
  if (!n || !query_count) return;
  if (!block || !queries || !scores) bad("null scoring buffer");
  if (score_stride<n) bad("score stride shorter than count");
  const size_t m=coded_dimensions();
  for (size_t i=0;i<query_count;++i) {
    const auto& q=queries[i];
    layout.validate_weights(q.correction_weights.size(),config_.num_centroids);
    if (q.weights.size()!=m) bad("query weight shape mismatch");
    if (validate_query) score(nullptr,0,q,nullptr,TieredQKernel::scalar);
    if (q.offsets.size()!=m || q.centroid_scores.size()!=config_.num_centroids) bad("query shape mismatch");
  }
  if (kernel==TieredQKernel::automatic)
    kernel=kernel_available(TieredQKernel::avx512) ? TieredQKernel::avx512
         : kernel_available(TieredQKernel::avx2) ? TieredQKernel::avx2
         : kernel_available(TieredQKernel::neon) ? TieredQKernel::neon : TieredQKernel::scalar;
  if (!kernel_available(kernel)) bad("requested scoring kernel is unavailable");
  bool geometry=!queries[0].scan_groups.empty();
  for (size_t i=1;i<query_count && geometry;++i) geometry=same_scan_geometry(queries[0],queries[i]);
#ifdef DV_TIEREDQ_NEON
  if (kernel==TieredQKernel::neon && geometry) {
    const size_t stride=bytes_per_vector(), bulk=n/tieredq_detail::neon_block_docs*tieredq_detail::neon_block_docs;
    size_t batched=0;
    while (query_count-batched>=4) {
      tieredq_detail::scan_neon_batch<4>(block,n,stride,queries+batched,layout,
          config_.num_centroids,scores+batched*score_stride,score_stride);
      batched+=4;
    }
    if (query_count-batched>=3) {
      tieredq_detail::scan_neon_batch<3>(block,n,stride,queries+batched,layout,
          config_.num_centroids,scores+batched*score_stride,score_stride);
      batched+=3;
    } else if (query_count-batched>=2) {
      tieredq_detail::scan_neon_batch<2>(block,n,stride,queries+batched,layout,
          config_.num_centroids,scores+batched*score_stride,score_stride);
      batched+=2;
    }
    if (bulk<n) for (size_t i=0;i<batched;++i)
      score_planes(block+bulk*stride,n-bulk,queries[i],scores+i*score_stride+bulk,kernel,false);
    for (size_t i=batched;i<query_count;++i)
      score_planes(block,n,queries[i],scores+i*score_stride,kernel,false);
    return;
  }
#endif
  for (size_t i=0;i<query_count;++i)
    score_planes(block,n,queries[i],scores+i*score_stride,kernel,false);
}
void TieredQ::pack(const uint8_t* rows,size_t count,uint8_t* planes) const {
  const size_t stride=bytes_per_vector(), bytes=product((count+15)/16,16*stride);
  if (count && (!rows || !planes)) bad("null packing buffer");
  if (bytes) std::fill_n(planes,bytes,0);
  for (size_t i=0;i<count;++i) for (size_t b=0;b<stride;++b)
    planes[(i/16)*16*stride+b*16+i%16]=rows[i*stride+b];
}
void TieredQ::unpack(const uint8_t* planes,size_t count,uint8_t* rows) const {
  const size_t stride=bytes_per_vector(); product(count,stride);
  if (count && (!rows || !planes)) bad("null unpacking buffer");
  for (size_t i=0;i<count;++i) for (size_t b=0;b<stride;++b)
    rows[i*stride+b]=planes[(i/16)*16*stride+b*16+i%16];
}
void TieredQ::decode(const uint8_t* payload,size_t count,float* rows) const {
  const Layout layout{code_bytes()};
  if (count && (!payload || !rows)) bad("null decoding buffer");
  const size_t d=model_dimensions(),m=coded_dimensions(),stride=bytes_per_vector();
  const bool lifted=config_.metric==TieredQMetric::l2;
  std::vector<double> precise(m);
  std::vector<float> model_row(lifted ? d : 0);
  for (size_t i=0;i<count;++i) {
    const uint8_t* code=payload+i*stride; const unsigned cid=code[code_bytes()];
    if (cid>=config_.num_centroids) bad("invalid centroid id");
    const double factor=layout.factor(code);
    for (const auto& t:tiers_) for (uint32_t j=0;j<t.dimensions;++j) {
      const size_t r=t.start+j; const float level=levels_[t.bits][tieredq_detail::read_wide_code(code,t.bit_offset+j*t.bits,t.bits)];
      precise[r]=mean_[r]+factor*(double(sigma_[r])*level);
    }
    float* row=lifted ? model_row.data() : rows+i*d;
    for (size_t j=0;j<d;++j) {
      double residual=0; for (size_t r=0;r<m;++r) residual+=basis_value(j*m+r)*precise[r];
      row[j]=float(centroids_[size_t(cid)*d+j]+residual);
    }
    if (!lifted) continue;
    // Invert the lift: [x/s, .] = row * base_map^-1, and base_map^-1 is the transposed query map.
    const size_t inputs=config_.dimensions;
    for (size_t j=0;j<inputs;++j) {
      double sum=0; for (size_t k=0;k<d;++k) sum+=double(row[k])*lift_query_map_[j*d+k];
      rows[i*inputs+j]=float(sum*lift_scale_);
    }
  }
}
float TieredQ::basis_value(size_t index) const {
  if (basis_half_.empty()) return basis_[index];
  const size_t m=coded_dimensions();
  return tieredq_detail::half_decode(basis_half_[tieredq_detail::basis_panel_position(model_dimensions(),index/m,index%m)]);
}
size_t TieredQ::memory_bytes() const {
  size_t n=sizeof(*this)+tiers_.capacity()*sizeof(TieredQTier);
  for (const auto* v:{&centroids_,&basis_,&mean_,&sigma_}) n+=v->capacity()*sizeof(float);
  n+=basis_half_.capacity()*sizeof(uint16_t);
  n+=levels_.capacity()*sizeof(std::vector<float>);
  for (const auto& v:levels_) n+=v.capacity()*sizeof(float);
  n+=centroid_half_norms_.capacity()*sizeof(float)+scan_plan_.capacity()*sizeof(tieredq_detail::ScanGroupPlan);
  for (const auto* v:{&lift_base_map_,&lift_query_map_,&lift_base_panels_.data,&lift_query_panels_.data}) n+=v->capacity()*sizeof(float);
  return n+encoder_->bytes();
}
uint32_t TieredQ::model_id() const {
  const auto data=serialize(); uint32_t h=2166136261u;
  for (auto b:data) { h^=b; h*=16777619u; } return h;
}
} // namespace dv
