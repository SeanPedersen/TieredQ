// The fixed grid: SIMD lanes are independent grid points, with ordered coordinate reductions.
#pragma once
#include "tieredq_encode_matrix.h"
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <array>
#include <cmath>
#include <vector>

namespace dv::tieredq_detail {
constexpr size_t scale_grid_points=49,scale_grid_lanes=8;
constexpr size_t scale_grid_storage=(scale_grid_points+scale_grid_lanes-1)/scale_grid_lanes*scale_grid_lanes;
inline const std::array<double,scale_grid_storage>& scale_grid() {
  static const auto grid=[] {
    std::array<double,scale_grid_storage> values{};
    for (size_t point=0;point<scale_grid_points;++point)
      values[point]=std::pow(2.0,-1.0+2.0*double(point)/double(scale_grid_points-1));
    std::fill(values.begin()+scale_grid_points,values.end(),values[scale_grid_points-1]);
    return values;
  }();
  return grid;
}
// Strict comparison selects the lower code at a threshold, including repeated thresholds.
inline unsigned quantize_code(double value,const std::vector<double>& thresholds) {
  unsigned code=0;
  for (size_t step=(thresholds.size()+1)/2;step;step/=2) {
    const unsigned probe=code+unsigned(step);
    if (value>thresholds[probe-1]) code=probe;
  }
  return code;
}
template<class Model,class Work> void scale_grid_scalar(const Model& model,const Work& work,double* dots,double* norms) {
  for (size_t point=0;point<scale_grid_points;++point) {
    double dot=0,norm=0;
    for (const auto& tier:model.tiers) for (size_t j=0;j<tier.dimensions;++j) {
      const size_t r=tier.start+j;
      const unsigned code=quantize_code(work.z[r]*scale_grid()[point],work.thresholds[tier.bits]);
      const double reconstructed=double(model.levels[tier.bits][code])*model.sigma[r];
      dot+=work.v[r]*reconstructed; norm+=reconstructed*reconstructed;
    }
    dots[point]=dot; norms[point]=norm;
  }
}
#if defined(__aarch64__)
template<class Model,class Work> void scale_grid_point_scalar(const Model& model,const Work& work,size_t point,double& dot,double& norm) {
  dot=0; norm=0;
  for (const auto& tier:model.tiers) for (size_t j=0;j<tier.dimensions;++j) {
    const size_t r=tier.start+j;
    const unsigned code=quantize_code(work.z[r]*scale_grid()[point],work.thresholds[tier.bits]);
    const double reconstructed=double(model.levels[tier.bits][code])*model.sigma[r];
    dot+=work.v[r]*reconstructed; norm+=reconstructed*reconstructed;
  }
}
// Two grid points per vector; each lane keeps the scalar coordinate order, so only FMA rounding differs.
template<class Model,class Work> void scale_grid_neon(const Model& model,const Work& work,double* dots,double* norms) {
  for (size_t point=0;point+1<scale_grid_points;point+=2) {
    const float64x2_t grid=vld1q_f64(scale_grid().data()+point);
    float64x2_t dot=vdupq_n_f64(0.0),norm=vdupq_n_f64(0.0);
    for (const auto& tier:model.tiers) {
      const auto& thresholds=work.thresholds[tier.bits];
      const auto& levels=model.levels[tier.bits];
      for (size_t j=0;j<tier.dimensions;++j) {
        const size_t r=tier.start+j;
        const float64x2_t value=vmulq_f64(vdupq_n_f64(work.z[r]),grid);
        uint64x2_t code=vdupq_n_u64(0);
        for (size_t step=(thresholds.size()+1)/2;step;step/=2) {
          const size_t first=size_t(vgetq_lane_u64(code,0))+step-1;
          const size_t second=size_t(vgetq_lane_u64(code,1))+step-1;
          const float64x2_t threshold=vsetq_lane_f64(thresholds[second],vdupq_n_f64(thresholds[first]),1);
          code=vaddq_u64(code,vandq_u64(vcgtq_f64(value,threshold),vdupq_n_u64(step)));
        }
        const size_t first=size_t(vgetq_lane_u64(code,0)),second=size_t(vgetq_lane_u64(code,1));
        const float64x2_t level=vsetq_lane_f64(double(levels[second]),vdupq_n_f64(double(levels[first])),1);
        const float64x2_t reconstructed=vmulq_f64(level,vdupq_n_f64(model.sigma[r]));
        dot=vfmaq_f64(dot,vdupq_n_f64(work.v[r]),reconstructed);
        norm=vfmaq_f64(norm,reconstructed,reconstructed);
      }
    }
    vst1q_f64(dots+point,dot); vst1q_f64(norms+point,norm);
  }
  scale_grid_point_scalar(model,work,scale_grid_points-1,dots[scale_grid_points-1],norms[scale_grid_points-1]);
}
#endif
#ifdef TIEREDQ_ENCODING_X86
template<unsigned Bits,class Model,class Work>
__attribute__((target("avx512f"))) inline void scale_grid_small_avx512(
    const Model& model,const Work& work,const TieredQTier& tier,__m512d grid,__m512d& dot,__m512d& norm) {
  const double* levels=work.small_levels[Bits].data();
  const auto& thresholds=work.thresholds[Bits];
  const __m512d low=_mm512_loadu_pd(levels),high=_mm512_loadu_pd(levels+8);
  for (size_t j=0;j<tier.dimensions;++j) {
    const size_t r=tier.start+j;
    const __m512d value=_mm512_mul_pd(_mm512_set1_pd(work.z[r]),grid);
    __m512d level;
    if constexpr (Bits==1) {
      const __mmask8 greater=_mm512_cmp_pd_mask(value,_mm512_set1_pd(thresholds[0]),_CMP_GT_OQ);
      level=_mm512_mask_blend_pd(greater,_mm512_set1_pd(levels[0]),_mm512_set1_pd(levels[1]));
    } else {
      __m512i code=_mm512_setzero_si512();
      for (size_t t=0;t<(1u<<Bits)-1;++t) {
        const __mmask8 greater=_mm512_cmp_pd_mask(value,_mm512_set1_pd(thresholds[t]),_CMP_GT_OQ);
        code=_mm512_mask_add_epi64(code,greater,code,_mm512_set1_epi64(1));
      }
      if constexpr (Bits<4) level=_mm512_permutexvar_pd(code,low);
      else level=_mm512_permutex2var_pd(low,code,high);
    }
    const __m512d reconstructed=_mm512_mul_pd(level,_mm512_set1_pd(model.sigma[r]));
    dot=_mm512_add_pd(dot,_mm512_mul_pd(_mm512_set1_pd(work.v[r]),reconstructed));
    norm=_mm512_add_pd(norm,_mm512_mul_pd(reconstructed,reconstructed));
  }
}
template<unsigned Bits,class Model,class Work>
__attribute__((target("avx512f"))) inline void scale_grid_wide_avx512(
    const Model& model,const Work& work,const TieredQTier& tier,__m512d grid,__m512d& dot,__m512d& norm) {
  const double* thresholds=work.thresholds[Bits].data();
  constexpr size_t four_step=1u<<(Bits-3),eight_step=four_step/2,sixteen_step=eight_step/2;
  const __m512d four=_mm512_setr_pd(thresholds[four_step-1],thresholds[3*four_step-1],
      thresholds[5*four_step-1],thresholds[7*four_step-1],0,0,0,0);
  const __m512d eight=_mm512_setr_pd(thresholds[eight_step-1],thresholds[3*eight_step-1],
      thresholds[5*eight_step-1],thresholds[7*eight_step-1],thresholds[9*eight_step-1],
      thresholds[11*eight_step-1],thresholds[13*eight_step-1],thresholds[15*eight_step-1]);
  const __m512d low=_mm512_setr_pd(thresholds[sixteen_step-1],thresholds[3*sixteen_step-1],
      thresholds[5*sixteen_step-1],thresholds[7*sixteen_step-1],thresholds[9*sixteen_step-1],
      thresholds[11*sixteen_step-1],thresholds[13*sixteen_step-1],thresholds[15*sixteen_step-1]);
  const __m512d high=_mm512_setr_pd(thresholds[17*sixteen_step-1],thresholds[19*sixteen_step-1],
      thresholds[21*sixteen_step-1],thresholds[23*sixteen_step-1],thresholds[25*sixteen_step-1],
      thresholds[27*sixteen_step-1],thresholds[29*sixteen_step-1],thresholds[31*sixteen_step-1]);
  // Depth 5 (32 thresholds) and up to 128 levels also stay in registers; only 8-bit tiers gather beyond depth 5.
  constexpr size_t thirty_two_step=sixteen_step/2>0 ? sixteen_step/2 : 1;
  constexpr size_t lanes=8,depth_five_vectors=4;
  __m512d depth_five[depth_five_vectors]{};
  if constexpr (Bits>=6)
    for (size_t part=0;part<depth_five_vectors;++part) {
      alignas(64) double values[lanes];
      for (size_t i=0;i<lanes;++i) values[i]=thresholds[(2*(part*lanes+i)+1)*thirty_two_step-1];
      depth_five[part]=_mm512_load_pd(values);
    }
  constexpr size_t register_levels=Bits<=7 ? size_t(1)<<Bits : 0,level_vectors=register_levels/16;
  __m512 levels[level_vectors>0 ? level_vectors : 1]{};
  for (size_t part=0;part<level_vectors;++part) levels[part]=_mm512_loadu_ps(model.levels[Bits].data()+16*part);
  for (size_t j=0;j<tier.dimensions;++j) {
    const size_t r=tier.start+j;
    const __m512d value=_mm512_mul_pd(_mm512_set1_pd(work.z[r]),grid);
    __m512i code=_mm512_setzero_si512();
    // The first five binary-search levels fit in registers; later levels retain exact gathers.
#pragma GCC unroll 8
    for (unsigned depth=0;depth<Bits;++depth) {
      const size_t step=1u<<(Bits-1-depth);
      const __m512i node=_mm512_srl_epi64(code,_mm_cvtsi32_si128(Bits-depth));
      __m512d threshold;
      if (depth==0) threshold=_mm512_set1_pd(thresholds[step-1]);
      else if (depth==1) threshold=_mm512_mask_blend_pd(_mm512_cmp_epi64_mask(code,_mm512_setzero_si512(),_MM_CMPINT_GT),
          _mm512_set1_pd(thresholds[step-1]),_mm512_set1_pd(thresholds[3*step-1]));
      else if (depth==2) threshold=_mm512_permutexvar_pd(node,four);
      else if (depth==3) threshold=_mm512_permutexvar_pd(node,eight);
      else if (depth==4) threshold=_mm512_permutex2var_pd(low,node,high);
      else if (depth==5) threshold=_mm512_mask_blend_pd(_mm512_cmp_epi64_mask(node,_mm512_set1_epi64(16),_MM_CMPINT_GE),
          _mm512_permutex2var_pd(depth_five[0],node,depth_five[1]),_mm512_permutex2var_pd(depth_five[2],node,depth_five[3]));
      else threshold=_mm512_i64gather_pd(_mm512_add_epi64(code,_mm512_set1_epi64(step-1)),thresholds,sizeof(double));
      const __mmask8 greater=_mm512_cmp_pd_mask(value,threshold,_CMP_GT_OQ);
      code=_mm512_mask_add_epi64(code,greater,code,_mm512_set1_epi64(step));
    }
    __m512d level;
    if constexpr (register_levels>0) {
      // Each permute covers 32 levels; blends on the higher code bits pick the right block.
      const __m512i indices=_mm512_castsi256_si512(_mm512_cvtepi64_epi32(code));
      __m512 blocks[level_vectors/2];
      for (size_t part=0;part<level_vectors/2;++part) blocks[part]=_mm512_permutex2var_ps(levels[2*part],indices,levels[2*part+1]);
      __m512 values=blocks[0];
      if constexpr (Bits>=6) {
        const __mmask8 odd_block=_mm512_test_epi64_mask(code,_mm512_set1_epi64(32));
        values=_mm512_mask_blend_ps(odd_block,blocks[0],blocks[1]);
        if constexpr (Bits==7) values=_mm512_mask_blend_ps(_mm512_test_epi64_mask(code,_mm512_set1_epi64(64)),values,
            _mm512_mask_blend_ps(odd_block,blocks[2],blocks[3]));
      }
      level=_mm512_cvtps_pd(_mm512_castps512_ps256(values));
    } else level=_mm512_cvtps_pd(_mm512_i64gather_ps(code,model.levels[Bits].data(),sizeof(float)));
    const __m512d reconstructed=_mm512_mul_pd(level,_mm512_set1_pd(model.sigma[r]));
    dot=_mm512_add_pd(dot,_mm512_mul_pd(_mm512_set1_pd(work.v[r]),reconstructed));
    norm=_mm512_add_pd(norm,_mm512_mul_pd(reconstructed,reconstructed));
  }
}
template<class Model,class Work>
__attribute__((target("avx512f"))) void scale_grid_avx512(const Model& model,const Work& work,double* dots,double* norms) {
  constexpr size_t lanes=8;
  for (size_t point=0;point<scale_grid_points;point+=lanes) {
    const __m512d grid=_mm512_loadu_pd(scale_grid().data()+point);
    __m512d dot=_mm512_setzero_pd(),norm=_mm512_setzero_pd();
    for (const auto& tier:model.tiers) {
      switch (tier.bits) {
        case 1: scale_grid_small_avx512<1>(model,work,tier,grid,dot,norm); continue;
        case 2: scale_grid_small_avx512<2>(model,work,tier,grid,dot,norm); continue;
        case 3: scale_grid_small_avx512<3>(model,work,tier,grid,dot,norm); continue;
        case 4: scale_grid_small_avx512<4>(model,work,tier,grid,dot,norm); continue;
        case 5: scale_grid_wide_avx512<5>(model,work,tier,grid,dot,norm); continue;
        case 6: scale_grid_wide_avx512<6>(model,work,tier,grid,dot,norm); continue;
        case 7: scale_grid_wide_avx512<7>(model,work,tier,grid,dot,norm); continue;
        case 8: scale_grid_wide_avx512<8>(model,work,tier,grid,dot,norm); continue;
        // Fine tiers (#26).
        case 9: scale_grid_wide_avx512<9>(model,work,tier,grid,dot,norm); continue;
        case 10: scale_grid_wide_avx512<10>(model,work,tier,grid,dot,norm); continue;
        case 11: scale_grid_wide_avx512<11>(model,work,tier,grid,dot,norm); continue;
        case 12: scale_grid_wide_avx512<12>(model,work,tier,grid,dot,norm); continue;
      }

    }
    _mm512_storeu_pd(dots+point,dot); _mm512_storeu_pd(norms+point,norm);
  }
}
template<class Model,class Work>
__attribute__((target("avx2"))) void scale_grid_avx2(const Model& model,const Work& work,double* dots,double* norms) {
  constexpr size_t lanes=4;
  for (size_t point=0;point<scale_grid_points;point+=lanes) {
    const __m256d grid=_mm256_loadu_pd(scale_grid().data()+point);
    __m256d dot=_mm256_setzero_pd(),norm=_mm256_setzero_pd();
    for (const auto& tier:model.tiers) for (size_t j=0;j<tier.dimensions;++j) {
      const size_t r=tier.start+j;
      const __m256d value=_mm256_mul_pd(_mm256_set1_pd(work.z[r]),grid);
      __m256d code=_mm256_setzero_pd();
      const auto& thresholds=work.thresholds[tier.bits];
      for (size_t step=(thresholds.size()+1)/2;step;step/=2) {
        const __m128i probe=_mm256_cvttpd_epi32(_mm256_add_pd(code,_mm256_set1_pd(step-1)));
        const __m256d threshold=_mm256_i32gather_pd(thresholds.data(),probe,sizeof(double));
        const __m256d greater=_mm256_cmp_pd(value,threshold,_CMP_GT_OQ);
        code=_mm256_add_pd(code,_mm256_and_pd(greater,_mm256_set1_pd(step)));
      }
      const __m256d level=_mm256_cvtps_pd(_mm_i32gather_ps(model.levels[tier.bits].data(),_mm256_cvttpd_epi32(code),sizeof(float)));
      const __m256d reconstructed=_mm256_mul_pd(level,_mm256_set1_pd(model.sigma[r]));
      dot=_mm256_add_pd(dot,_mm256_mul_pd(_mm256_set1_pd(work.v[r]),reconstructed));
      norm=_mm256_add_pd(norm,_mm256_mul_pd(reconstructed,reconstructed));
    }
    _mm256_storeu_pd(dots+point,dot); _mm256_storeu_pd(norms+point,norm);
  }
}
#endif
template<class Model,class Work> void scale_grid_terms(const Model& model,const Work& work,double* dots,double* norms,
                                                    TieredQKernel kernel=TieredQKernel::automatic) {
  switch (encoding_kernel(kernel)) {
#if defined(__aarch64__)
    case TieredQKernel::neon: return scale_grid_neon(model,work,dots,norms);
#endif
#ifdef TIEREDQ_ENCODING_X86
    case TieredQKernel::avx512: return scale_grid_avx512(model,work,dots,norms);
    case TieredQKernel::avx2: return scale_grid_avx2(model,work,dots,norms);
#endif
    default: return scale_grid_scalar(model,work,dots,norms);
  }
}
} // namespace dv::tieredq_detail
