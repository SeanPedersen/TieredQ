// The serving binary16 PCA basis in rank panels: [rank / 16][input][rank % 16], ranks zero-padded to whole panels.
// A query streams each panel contiguously (the input-major order strides by m and stalls the prefetcher well below
// cache bandwidth), and a query tile shares every basis vector it loads. Files keep the input-major d x m order.
// Projections keep each output's FP32 sequence (inputs ascending, multiply then add), so the layout never changes
// a weight: out[q][r] = sum_i x_q[i] * c(i,r), with c = basis or, folded, basis * sigma[r] rounded first.
#pragma once
#include "tieredq_layout.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define DV_TIEREDQ_BASIS_X86 1
#endif

namespace dv::tieredq_detail {
constexpr size_t basis_panel_ranks=16;
// Largest query tile one projection pass shares the basis across.
constexpr size_t basis_query_tile=8;

inline size_t basis_panel_count(size_t ranks) { return (ranks+basis_panel_ranks-1)/basis_panel_ranks; }
inline size_t basis_panel_size(size_t inputs,size_t ranks) { return basis_panel_count(ranks)*inputs*basis_panel_ranks; }
inline size_t basis_panel_position(size_t inputs,size_t input,size_t rank) {
  return (rank/basis_panel_ranks*inputs+input)*basis_panel_ranks+rank%basis_panel_ranks;
}
inline std::vector<uint16_t> basis_panels_from_rows(const std::vector<uint16_t>& rows,size_t inputs,size_t ranks) {
  std::vector<uint16_t> panels(basis_panel_size(inputs,ranks),0);
  for (size_t i=0;i<inputs;++i) for (size_t r=0;r<ranks;++r) panels[basis_panel_position(inputs,i,r)]=rows[i*ranks+r];
  return panels;
}
inline std::vector<uint16_t> basis_rows_from_panels(const std::vector<uint16_t>& panels,size_t inputs,size_t ranks) {
  std::vector<uint16_t> rows(inputs*ranks);
  for (size_t i=0;i<inputs;++i) for (size_t r=0;r<ranks;++r) rows[i*ranks+r]=panels[basis_panel_position(inputs,i,r)];
  return rows;
}
inline bool basis_panel_padding_zero(const std::vector<uint16_t>& panels,size_t inputs,size_t ranks) {
  for (size_t r=ranks;r<basis_panel_count(ranks)*basis_panel_ranks;++r)
    for (size_t i=0;i<inputs;++i) if (panels[basis_panel_position(inputs,i,r)]) return false;
  return true;
}

struct BasisProjection {
  const uint16_t* panels;
  size_t inputs,ranks;
  const float* sigma;  // folded into the coefficient when not null
  const float* vectors;
  size_t vector_stride;
  float* out;
  size_t out_stride;
};

#ifdef DV_TIEREDQ_BASIS_X86
// Queries x Panels accumulators stay in registers over all inputs; Panels counts the live panels of this block.
template<size_t Queries,size_t Panels>
__attribute__((target("avx512f"))) inline void project_panel_block_avx512(const BasisProjection& p,size_t panel) {
  __m512 acc[Queries][Panels],scale[Panels];
  for (size_t b=0;b<Panels;++b) {
    const size_t rank=(panel+b)*basis_panel_ranks;
    const auto mask=__mmask16((1u<<std::min(basis_panel_ranks,p.ranks-rank))-1);
    scale[b]=p.sigma ? _mm512_maskz_loadu_ps(mask,p.sigma+rank) : _mm512_setzero_ps();
    for (size_t q=0;q<Queries;++q) acc[q][b]=_mm512_setzero_ps();
  }
  const uint16_t* base=p.panels+panel*p.inputs*basis_panel_ranks;
  for (size_t i=0;i<p.inputs;++i) {
    __m512 x[Queries];
    for (size_t q=0;q<Queries;++q) x[q]=_mm512_set1_ps(p.vectors[q*p.vector_stride+i]);
    for (size_t b=0;b<Panels;++b) {
      auto c=_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(base+(b*p.inputs+i)*basis_panel_ranks)));
      if (p.sigma) c=_mm512_mul_ps(c,scale[b]);
      for (size_t q=0;q<Queries;++q) acc[q][b]=_mm512_add_ps(acc[q][b],_mm512_mul_ps(x[q],c));
    }
  }
  for (size_t b=0;b<Panels;++b) {
    const size_t rank=(panel+b)*basis_panel_ranks;
    const auto mask=__mmask16((1u<<std::min(basis_panel_ranks,p.ranks-rank))-1);
    for (size_t q=0;q<Queries;++q) _mm512_mask_storeu_ps(p.out+q*p.out_stride+rank,mask,acc[q][b]);
  }
}
// Panels per block leave room for the broadcasts and folded scales within 32 registers.
template<size_t Queries> constexpr size_t avx512_block_panels=Queries>=8 ? 3 : Queries>=4 ? 5 : Queries>=2 ? 8 : 12;
template<size_t Queries,size_t Panels=avx512_block_panels<Queries>>
__attribute__((target("avx512f"))) inline void project_panel_tail_avx512(const BasisProjection& p,size_t panel,size_t live) {
  if constexpr (Panels>1) if (live<Panels) return project_panel_tail_avx512<Queries,Panels-1>(p,panel,live);
  project_panel_block_avx512<Queries,Panels>(p,panel);
}
template<size_t Queries>
__attribute__((target("avx512f"))) inline void project_queries_avx512(const BasisProjection& p) {
  constexpr size_t block=avx512_block_panels<Queries>;
  const size_t panels=basis_panel_count(p.ranks);
  size_t panel=0;
  for (;panel+block<=panels;panel+=block) project_panel_block_avx512<Queries,block>(p,panel);
  if (panel<panels) project_panel_tail_avx512<Queries>(p,panel,panels-panel);
}

// AVX2 holds a panel as two halves and reloads sigma instead of keeping it in the 16 registers.
template<size_t Queries,size_t Panels>
__attribute__((target("avx2,f16c"))) inline void project_panel_block_avx2(const BasisProjection& p,size_t panel) {
  constexpr size_t lanes=8,halves=basis_panel_ranks/lanes;
  __m256 acc[Queries][Panels*halves];
  for (auto& row:acc) for (auto& value:row) value=_mm256_setzero_ps();
  alignas(32) float sigma[Panels*basis_panel_ranks]={};
  if (p.sigma) for (size_t r=panel*basis_panel_ranks;r<std::min(p.ranks,(panel+Panels)*basis_panel_ranks);++r)
    sigma[r-panel*basis_panel_ranks]=p.sigma[r];
  const uint16_t* base=p.panels+panel*p.inputs*basis_panel_ranks;
  for (size_t i=0;i<p.inputs;++i) {
    __m256 x[Queries];
    for (size_t q=0;q<Queries;++q) x[q]=_mm256_set1_ps(p.vectors[q*p.vector_stride+i]);
    for (size_t h=0;h<Panels*halves;++h) {
      const size_t b=h/halves,offset=h%halves*lanes;
      auto c=_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(base+(b*p.inputs+i)*basis_panel_ranks+offset)));
      if (p.sigma) c=_mm256_mul_ps(c,_mm256_load_ps(sigma+h*lanes));
      for (size_t q=0;q<Queries;++q) acc[q][h]=_mm256_add_ps(acc[q][h],_mm256_mul_ps(x[q],c));
    }
  }
  for (size_t q=0;q<Queries;++q) {
    alignas(32) float values[Panels*basis_panel_ranks];
    for (size_t h=0;h<Panels*halves;++h) _mm256_store_ps(values+h*lanes,acc[q][h]);
    const size_t rank=panel*basis_panel_ranks,live=std::min(p.ranks-rank,Panels*basis_panel_ranks);
    std::copy_n(values,live,p.out+q*p.out_stride+rank);
  }
}
template<size_t Queries> constexpr size_t avx2_block_panels=Queries>=4 ? 1 : Queries>=2 ? 2 : 4;
template<size_t Queries,size_t Panels=avx2_block_panels<Queries>>
__attribute__((target("avx2,f16c"))) inline void project_panel_tail_avx2(const BasisProjection& p,size_t panel,size_t live) {
  if constexpr (Panels>1) if (live<Panels) return project_panel_tail_avx2<Queries,Panels-1>(p,panel,live);
  project_panel_block_avx2<Queries,Panels>(p,panel);
}
template<size_t Queries>
__attribute__((target("avx2,f16c"))) inline void project_queries_avx2(const BasisProjection& p) {
  constexpr size_t block=avx2_block_panels<Queries>;
  const size_t panels=basis_panel_count(p.ranks);
  size_t panel=0;
  for (;panel+block<=panels;panel+=block) project_panel_block_avx2<Queries,block>(p,panel);
  if (panel<panels) project_panel_tail_avx2<Queries>(p,panel,panels-panel);
}
#endif

inline void project_queries_scalar(const BasisProjection& p,size_t rows) {
  for (size_t q=0;q<rows;++q) {
    float* out=p.out+q*p.out_stride;
    std::fill_n(out,p.ranks,0.0f);
    for (size_t i=0;i<p.inputs;++i) {
      const float x=p.vectors[q*p.vector_stride+i];
      for (size_t r=0;r<p.ranks;++r) {
        float c=half_decode(p.panels[basis_panel_position(p.inputs,i,r)]);
        if (p.sigma) c=c*p.sigma[r];
        out[r]=out[r]+x*c;
      }
    }
  }
}

// Projects `rows` vectors in query tiles of up to basis_query_tile, writing out[row * out_stride + rank].
inline void project_basis_panels(BasisProjection p,size_t rows) {
#ifdef DV_TIEREDQ_BASIS_X86
  const bool avx512=__builtin_cpu_supports("avx512f"),avx2=__builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
  if (avx512 || avx2) {
    while (rows) {
      const size_t tile=rows>=8 && avx512 ? 8 : rows>=4 ? 4 : rows>=2 ? 2 : 1;
      if (avx512) {
        if (tile==8) project_queries_avx512<8>(p); else if (tile==4) project_queries_avx512<4>(p);
        else if (tile==2) project_queries_avx512<2>(p); else project_queries_avx512<1>(p);
      } else {
        if (tile==4) project_queries_avx2<4>(p); else if (tile==2) project_queries_avx2<2>(p); else project_queries_avx2<1>(p);
      }
      p.vectors+=tile*p.vector_stride; p.out+=tile*p.out_stride; rows-=tile;
    }
    return;
  }
#endif
  project_queries_scalar(p,rows);
}
} // namespace dv::tieredq_detail
