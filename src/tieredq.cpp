// TieredQ training, persistence, and shared scalar/SIMD scan kernels.
#include "tieredq/tieredq.h"
#include "tieredq_layout.h"
#include "tieredq_io.h"
#include "tieredq_encode.h"
#include "tieredq_encode_float.h"
#include "tieredq_profile.h"
#include "tieredq_train_linalg.h"
#include "tieredq_train_kmeans.h"
#include "tieredq_train_stats.h"
#include "tieredq_train_residuals.h"
#include "tieredq_train_codebook.h"
#include "tieredq_train_codebook_data.h"
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
#include <queue>
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
template<class F> TieredQPanels pack_panels(size_t depth, size_t columns, F at) {
  TieredQPanels p; p.depth = depth; p.columns = columns;
  p.data.assign(product(depth, p.padded_columns()), 0);
  for (size_t c = 0; c < columns; ++c) for (size_t k = 0; k < depth; ++k) p.data[(c / PANEL * depth + k) * PANEL + c % PANEL] = at(k, c);
  return p;
}
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
void nearest_centroids(const float* x, size_t rows, size_t d, const std::vector<float>& centroids, size_t k, uint32_t* ids) {
  const auto panels = pack_panels(d, k, [&](size_t i, size_t c) { return centroids[c * d + i]; });
  const auto half = half_norms(centroids, k, d);
  const size_t width = panels.padded_columns(), blocks = (rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
#pragma omp parallel for schedule(dynamic) if(blocks > 1)
  for (size_t blk = 0; blk < blocks; ++blk) {
    const size_t r0 = blk * BLOCK_ROWS, n = std::min(BLOCK_ROWS, rows - r0);
    std::vector<float> dots(n * width);
    float_matrix(x + r0 * d, n, d, panels, dots.data(), width);
    for (size_t r = 0; r < n; ++r) ids[r0 + r] = argmin_centroid(dots.data() + r * width, half.data(), k);
  }
}
// Lower-triangle covariance in column-major LAPACK layout from a d x rows (transposed) matrix.
std::vector<double> covariance(const std::vector<double>& transposed, int d, size_t rows) {
  std::vector<double> cov(product(d, d), 0); const double alpha = 1.0 / double(rows - 1);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < d; ++i) for (int j = 0; j <= i; ++j) {
    const double* a = transposed.data() + size_t(i) * rows; const double* b = transposed.data() + size_t(j) * rows; double s = 0;
#pragma omp simd reduction(+:s)
    for (size_t r = 0; r < rows; ++r) s += a[r] * b[r];
    cov[size_t(j) * d + i] = cov[size_t(i) * d + j] = s * alpha;
  }
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
uint32_t read_code(const uint8_t* data, uint32_t offset, unsigned bits) {
  uint32_t value=data[offset/8]>> (offset%8);
  if (offset%8+bits>8) value |= uint32_t(data[offset/8+1]) << (8-offset%8);
  return value & ((1u<<bits)-1);
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
  if (!max_bits || max_bits>8) bad("max bits per coordinate must be in [1, 8]");
}
TieredQ TieredQ::train(TieredQConfig config,const float* sample,size_t rows,uint64_t seed) {
  config.validate(); if (rows<std::max<size_t>(2,size_t(config.num_centroids)*2) || rows>INT32_MAX) bad("training needs at least 2*K rows and at most INT32_MAX rows");
  const int d=int(config.dimensions), n=int(rows); const unsigned k=config.num_centroids;
  tieredq_detail::StageProfile profile;
  const auto& linalg=tieredq_detail::TrainingLinalg::get();
  finite_rows(sample,product(rows,d)); TieredQ c; c.config_=config;
  tieredq_detail::TrainingRng rng(seed); auto initial=rng.choice(rows,k);
  c.centroids_.resize(product(k,d));
  for (unsigned j=0;j<k;++j) std::copy_n(sample+initial[j]*d,d,c.centroids_.data()+size_t(j)*d);
  std::vector<uint32_t> ids(rows);
  profile.mark("init");
  constexpr int kmeans_iterations=20;
  const int iterations=k==1 ? 1 : kmeans_iterations;
  for (int it=0;it<iterations;++it) {
    if (k>1) nearest_centroids(sample,rows,d,c.centroids_,k,ids.data());
    profile.mark("kmeans_assign");
    tieredq_detail::update_centroids(sample,rows,d,ids.data(),c.centroids_);
    profile.mark("kmeans_update");
  }
  const bool spectral_stats=!std::getenv("DV_TIEREDQ_REFERENCE_STATS");
  std::vector<double> residual,mean,cov;
  if (k>1) nearest_centroids(sample,rows,d,c.centroids_,k,ids.data());
  profile.mark("kmeans_assign");
  if (spectral_stats && linalg.syrk) {
    auto centered=tieredq_detail::centered_residuals(sample,rows,d,ids.data(),c.centroids_);
    mean=std::move(centered.mean);
    profile.mark("residual");
    cov=linalg.covariance_centered(centered.values,d,n);
  } else {
    residual.resize(product(rows,d)); mean.assign(d,0);
    for (size_t i=0;i<rows;++i) {
      const auto id=ids[i];
      for (int j=0;j<d;++j) { double v=double(sample[i*d+j])-c.centroids_[size_t(id)*d+j]; residual[i*d+j]=v; mean[j]+=v; }
    }
    for (int j=0;j<d;++j) mean[j]/=rows;
    if (linalg.syrk) cov=linalg.covariance(residual,mean,d,n,profile);
    else {
      std::vector<double> transposed(residual.size());
      for (size_t i=0;i<rows;++i) for (int j=0;j<d;++j) transposed[size_t(j)*rows+i]=residual[i*d+j]-mean[j];
      profile.mark("residual");
      cov=covariance(transposed,d,rows);
    }
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
  c.levels_.resize(9); std::array<double,9> error{}; error[0]=1;
  for (unsigned b=1;b<=8;++b) { c.levels_[b]=tieredq_detail::gaussian_codebooks()[b]; error[b]=tieredq_detail::distortion(c.levels_[b]); }
  std::vector<unsigned> bits(d,0);
  struct Gain { double value; int rank; bool operator<(const Gain& b) const { return value<b.value || (value==b.value && rank>b.rank); } };
  std::priority_queue<Gain> gains;
  auto gain=[&](int rank) { unsigned b=bits[rank]; return std::max(0.0,eigenvalues[d-1-rank])*(error[b]-error[b+1]); };
  for (int j=0;j<d;++j) gains.push({gain(j),j});
  auto cap=[&](int rank) {
    return uint32_t(rank)<config.wide_dims ? config.max_bits : std::min<uint32_t>(config.max_bits,6);
  };
  for (uint32_t used=0;used<config.code_bits;++used) {
    if (gains.empty()) bad("code_bits exceeds the allocation cap");
    int j=gains.top().rank; gains.pop(); ++bits[j];
    if (bits[j]<cap(j)) gains.push({gain(j),j});
  }
  profile.mark("bit_allocation");
  // Group equal precisions; retain source PCA ranks explicitly in the unrotated basis.
  std::vector<int> ranks; uint32_t offset=0;
  for (int b=8;b>=1;--b) {
    const auto start=ranks.size(); for (int j=0;j<d;++j) if (bits[j]==unsigned(b)) ranks.push_back(j);
    if (ranks.size()!=start) {
      const uint32_t count=uint32_t(ranks.size()-start);
      c.tiers_.push_back({uint32_t(start),count,uint32_t(b),0,offset}); offset+=count*b;
    }
  }
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
  if (spectral_stats) {
    tieredq_detail::spectral_projection_stats(eigenvalues,ranks,c.tiers_,rotations,mean,c.basis_,rows,c.mean_,c.sigma_);
  } else {
    // Residual covariance was centered; restore the mean before fitting the projected mean.
    std::vector<float> rfloat(residual.size());
    for (size_t i=0;i<rows;++i) for (int j=0;j<d;++j) rfloat[i*d+j]=float(residual[i*d+j]);
    std::vector<float> projected(product(rows,m)); multiply(rfloat.data(),n,d,c.basis_.data(),m,projected.data());
    tieredq_detail::projection_stats(projected,rows,m,c.mean_,c.sigma_);
  }
  profile.mark("projection_stats");
  c.prepare_encoder();
  profile.mark("prepare_encoder");
  c.validate_model(); return c;
}

void TieredQ::prepare_encoder() {
  const size_t d=config_.dimensions,m=coded_dimensions(),k=config_.num_centroids;
  encode_panels_=pack_panels(d,m+k,[&](size_t j,size_t r) {
    return r<m ? basis_[j*m+r] : centroids_[(r-m)*d+j];
  });
  const auto projection=pack_panels(d,m,[&](size_t j,size_t r) { return basis_[j*m+r]; });
  const std::vector<double> centers(centroids_.begin(),centroids_.end());
  centroid_projections_.resize(product(k,projection.padded_columns()));
  tieredq_detail::ordered_matrix(centers.data(),k,projection,centroid_projections_.data());
  exact_centroid_half_norms_=exact_half_norms(centroids_,k,d);
  centroid_half_norms_.assign(exact_centroid_half_norms_.begin(),exact_centroid_half_norms_.end());
  tieredq_detail::build_scan_plan(tiers_,scan_plan_);
}
// One fused GEMM yields the basis projection [0, m) and centroid dots [m, m+K) per row;
// the residual projection is x*basis - c*basis.
void TieredQ::encode_block(const float* rows,size_t count,uint8_t* out,float* corrections,tieredq_detail::StageProfile* profile) const {
  const Layout layout{code_bytes()};
  const tieredq_detail::EncodingModel model{config_.dimensions,centroids_,mean_,sigma_,levels_,tiers_};
  const tieredq_detail::EncoderState encoder{encode_panels_,centroid_projections_,exact_centroid_half_norms_};
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
  finite_rows(vector,config_.dimensions); TieredQQuery q; const size_t m=coded_dimensions(),d=config_.dimensions;
  std::vector<float> transformed(m,0);
  for (size_t i=0;i<d;++i) for (size_t j=0;j<m;++j) transformed[j]+=vector[i]*basis_[i*m+j];
  q.offsets.resize(m); q.weights.resize(m);
  for (const auto& t:tiers_) for (uint32_t j=0;j<t.dimensions;++j) {
    const size_t rank=t.start+j; q.offsets[rank]=q.lut.size();
    q.weights[rank]=transformed[rank]*sigma_[rank];
    for (float level:levels_[t.bits]) q.lut.push_back(q.weights[rank]*level);
  }
  q.centroid_scores.assign(config_.num_centroids,0); q.correction_weights.assign(config_.num_centroids,0);
  // Quantization removes the projected residual mean; restore it in the score.
  float correction=0; for (size_t j=0;j<m;++j) correction+=transformed[j]*mean_[j];
  for (uint32_t id=0;id<config_.num_centroids;++id) {
    float s=0; for (size_t j=0;j<d;++j) s+=vector[j]*centroids_[size_t(id)*d+j]; q.centroid_scores[id]=s+correction;
    const float norm=2*centroid_half_norms_[id]; q.correction_weights[id]=norm>0 ? s/norm : 0;
  }
  tieredq_detail::build_scan_groups(tiers_,q);
  return q;
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
      sum+=q.lut[q.offsets[t.start+j]+read_code(doc,t.bit_offset+j*t.bits,t.bits)];
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
  config_.validate(); const size_t d=config_.dimensions,m=sigma_.size();
  if (!m || m>d || tiers_.empty() || tiers_.size()>8 || levels_.size()!=9)
    bad("invalid model layout");
  if (centroids_.size()!=product(config_.num_centroids,d) || basis_.size()!=product(d,m) || mean_.size()!=m) bad("invalid model array shape");
  uint32_t rank=0; uint64_t offset=0;
  for (size_t i=0;i<tiers_.size();++i) {
    const auto& t=tiers_[i];
    if (t.start!=rank || !t.dimensions || t.dimensions>m-rank || !t.bits || t.bits>8 || t.scale_group!=0 || t.bit_offset!=offset) bad("invalid tier descriptor");
    rank+=t.dimensions; offset+=uint64_t(t.dimensions)*t.bits;
  }
  if (rank!=m || offset!=config_.code_bits) bad("tier budget mismatch");
  if (!levels_[0].empty()) bad("invalid zero-bit table");
  for (unsigned b=1;b<=8;++b) if (levels_[b].size()!=(size_t(1)<<b) || !std::is_sorted(levels_[b].begin(),levels_[b].end())) bad("invalid Lloyd-Max table");
  for (float s:sigma_) if (!(s>0) || !std::isfinite(s)) bad("invalid sigma");
  for (const auto* a:{&centroids_,&basis_,&mean_}) finite_rows(a->data(),a->size());
  for (const auto& a:levels_) finite_rows(a.data(),a.size());
}
std::vector<uint8_t> TieredQ::serialize() const {
  validate_model(); Writer w;
  w.u32(config_.dimensions); w.u32(config_.code_bits); w.u32(config_.num_centroids);
  w.u32(config_.rotate_tiers); w.u32(model_format); w.u32(uint32_t(code_bytes())); w.u32(uint32_t(bytes_per_vector()));
  w.u32(uint32_t(tiers_.size()));
  for (const auto& t:tiers_) { w.u32(t.start); w.u32(t.dimensions); w.u32(t.bits); w.u32(t.scale_group); w.u32(t.bit_offset); }
  w.floats(centroids_);
  for (const auto* v:{&basis_,&mean_,&sigma_}) w.floats(*v);
  w.u32(uint32_t(levels_.size())); for (const auto& l:levels_) w.floats(l);
  return w.data;
}
TieredQ TieredQ::deserialize(const std::vector<uint8_t>& data) {
  Reader r{data}; TieredQ c;
  c.config_.dimensions=r.u32(); c.config_.code_bits=r.u32(); c.config_.num_centroids=r.u32();
  const auto rotate=r.u32(); if (rotate>1) bad("invalid rotation flag"); c.config_.rotate_tiers=rotate;
  if (r.u32()!=model_format) bad("unsupported codec format");
  c.config_.validate();
  auto cb=r.u32(),pb=r.u32(),nt=r.u32(); if (cb!=c.code_bytes() || nt==0 || nt>8) bad("invalid layout header");
  for (uint32_t i=0;i<nt;++i) c.tiers_.push_back({r.u32(),r.u32(),r.u32(),r.u32(),r.u32()});
  c.centroids_=r.floats();
  for (auto* v:{&c.basis_,&c.mean_,&c.sigma_}) *v=r.floats();
  if (r.u32()!=9) bad("table count mismatch"); for (int i=0;i<9;++i) c.levels_.push_back(r.floats());
  if (r.pos!=data.size() || pb!=c.bytes_per_vector()) bad("metadata length mismatch");
  c.validate_model(); c.prepare_encoder(); return c;
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
  const size_t d=config_.dimensions,m=coded_dimensions(),stride=bytes_per_vector();
  std::vector<double> precise(m);
  for (size_t i=0;i<count;++i) {
    const uint8_t* code=payload+i*stride; const unsigned cid=code[code_bytes()];
    if (cid>=config_.num_centroids) bad("invalid centroid id");
    const double factor=layout.factor(code);
    for (const auto& t:tiers_) for (uint32_t j=0;j<t.dimensions;++j) {
      const size_t r=t.start+j; const float level=levels_[t.bits][read_code(code,t.bit_offset+j*t.bits,t.bits)];
      precise[r]=mean_[r]+factor*(double(sigma_[r])*level);
    }
    for (size_t j=0;j<d;++j) {
      double residual=0; for (size_t r=0;r<m;++r) residual+=basis_[j*m+r]*precise[r];
      rows[i*d+j]=float(centroids_[size_t(cid)*d+j]+residual);
    }
  }
}
size_t TieredQ::memory_bytes() const {
  size_t n=sizeof(*this)+tiers_.capacity()*sizeof(TieredQTier);
  for (const auto* v:{&centroids_,&basis_,&mean_,&sigma_}) n+=v->capacity()*sizeof(float);
  n+=levels_.capacity()*sizeof(std::vector<float>);
  for (const auto& v:levels_) n+=v.capacity()*sizeof(float);
  n+=(encode_panels_.data.capacity()+centroid_half_norms_.capacity())*sizeof(float);
  n+=(centroid_projections_.capacity()+exact_centroid_half_norms_.capacity())*sizeof(double);
  return n;
}
uint32_t TieredQ::model_id() const {
  const auto data=serialize(); uint32_t h=2166136261u;
  for (auto b:data) { h^=b; h*=16777619u; } return h;
}
} // namespace dv
