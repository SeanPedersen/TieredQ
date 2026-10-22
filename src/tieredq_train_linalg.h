// Optional dynamically loaded training backend; otherwise use the linked LAPACK routines.
#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include "tieredq_profile.h"
#include "tieredq_train_covariance.h"
#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

extern "C" {
void dsyevd_(const char*,const char*,const int*,double*,const int*,double*,double*,const int*,int*,const int*,int*);
void dgeqrf_(const int*,const int*,double*,const int*,double*,double*,const int*,int*);
void dorgqr_(const int*,const int*,const int*,double*,const int*,const double*,double*,const int*,int*);
void dsyrk_(const char*,const char*,const int*,const int*,const double*,const double*,const int*,const double*,double*,const int*);
}

namespace dv::tieredq_detail {
class TrainingLinalg {
 public:
  decltype(&dsyevd_) eigen=&dsyevd_;
  decltype(&dgeqrf_) qr=&dgeqrf_;
  decltype(&dorgqr_) generate_q=&dorgqr_;
  decltype(&dsyrk_) syrk=nullptr;

  std::vector<double> covariance(const std::vector<double>& residual,const std::vector<double>& mean,
                                 int dimensions,int rows,StageProfile& profile) const {
    std::vector<double> centered(residual.size());
    for (int i=0;i<rows;++i) for (int j=0;j<dimensions;++j)
      centered[size_t(i)*dimensions+j]=residual[size_t(i)*dimensions+j]-mean[j];
    profile.mark("residual");
    return covariance_centered(centered,dimensions,rows);
  }

  std::vector<double> covariance_centered(const std::vector<double>& centered,int dimensions,int rows) const {
#if defined(__x86_64__) || defined(__i386__)
    constexpr int small_covariance_dimensions=200;
    if (dimensions<=small_covariance_dimensions && __builtin_cpu_supports("avx512f"))
      return small_covariance_avx512(centered,dimensions,rows);
#endif
    std::vector<double> result(size_t(dimensions)*dimensions);
    const char lower='L',normal='N'; const double alpha=1.0/double(rows-1),zero=0;
    syrk(&lower,&normal,&dimensions,&rows,&alpha,centered.data(),&dimensions,&zero,result.data(),&dimensions);
    return result;
  }

  static const TrainingLinalg& get() {
    static const TrainingLinalg selected=load();
    return selected;
  }

 private:
  static TrainingLinalg load() {
    TrainingLinalg result;
    const char* path=std::getenv("TIEREDQ_BLAS");
    if (!path || !*path) return result;
    const char* stage=std::getenv("TIEREDQ_BLAS_STAGE");
    const std::string selection=stage ? stage : "all";
    if (selection!="all" && selection!="covariance" && selection!="eigen" && selection!="qr")
      throw std::runtime_error("TieredQ: BLAS stage must be all, covariance, eigen or qr");
#if defined(__unix__) || defined(__APPLE__)
    int flags=RTLD_NOW|RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
    flags|=RTLD_DEEPBIND;
#endif
    // The backend must use LP64 Fortran integers. Keep it loaded for the cached function pointers.
    void* handle=dlopen(path,flags);
    if (!handle) throw std::runtime_error(std::string("TieredQ: cannot load BLAS: ")+dlerror());
    try {
      result.eigen=symbol<decltype(result.eigen)>(handle,"dsyevd_");
      result.qr=symbol<decltype(result.qr)>(handle,"dgeqrf_");
      result.generate_q=symbol<decltype(result.generate_q)>(handle,"dorgqr_");
      result.syrk=symbol<decltype(result.syrk)>(handle,"dsyrk_");
    } catch (...) { dlclose(handle); throw; }
    if (selection=="all") return result;
    if (selection!="covariance") result.syrk=nullptr;
    if (selection!="eigen") result.eigen=&dsyevd_;
    if (selection!="qr") { result.qr=&dgeqrf_; result.generate_q=&dorgqr_; }
    return result;
#else
    throw std::runtime_error("TieredQ: dynamic BLAS is unavailable on this platform");
#endif
  }
#if defined(__unix__) || defined(__APPLE__)
  template<class Function> static Function symbol(void* handle,const char* name) {
    auto value=reinterpret_cast<Function>(dlsym(handle,name));
    if (!value) throw std::runtime_error(std::string("TieredQ: BLAS is missing ")+name);
    return value;
  }
#endif
};
} // namespace dv::tieredq_detail
