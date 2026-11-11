// Metric transforms between input vectors and the inner-product model space (TieredQMetric).
// Cosine normalizes vectors. L2 lifts a row x to [x/s, -|x/s|^2/2] L and a query q to [q/s, 1] L^-T, so that
// lifted inner products equal (q.x - |x|^2/2)/s^2 and rank by distance. L is the lower Cholesky factor of the
// lifted query second moment, estimated with training rows as queries; it whitens the lifted queries (as the
// benchmark lift of #4 did). s is the largest training row norm.
#pragma once
#include "tieredq_io.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" {
void dpotrf_(const char*,const int*,double*,const int*,int*);
void dtrtri_(const char*,const char*,const int*,double*,const int*,int*);
}

namespace tq::tieredq_detail {
constexpr size_t lift_calibration_rows=32768;
// Lower Cholesky factor of moment + ridge * I in column-major order, or empty if it is not positive definite.
inline std::vector<double> lower_cholesky(std::vector<double> moment,int n,double ridge) {
  for (int i=0;i<n;++i) moment[size_t(i)*n+i]+=ridge;
  const char lower='L'; int info=0;
  dpotrf_(&lower,&n,moment.data(),&n,&info);
  if (info) return {};
  for (int j=0;j<n;++j) for (int i=0;i<j;++i) moment[size_t(j)*n+i]=0;
  return moment;
}
// Base map (row-major lower factor) from the lifted second moment (row-major, symmetric). Eigen is dsyevd; rank
// deficiency follows numpy.linalg.matrix_rank (largest eigenvalue * n * epsilon) and takes a float32 ridge.
template<class Eigen> std::vector<float> l2_base_map(const std::vector<double>& moment,size_t lifted,Eigen eigen) {
  const int n=int(lifted);
  double trace=0; for (size_t a=0;a<lifted;++a) trace+=moment[a*lifted+a];
  std::vector<double> values(lifted),scratch=moment;
  const char no_vectors='N',lower='L'; int lwork=-1,liwork=-1,info=0,iquery=0; double query=0;
  eigen(&no_vectors,&lower,&n,scratch.data(),&n,values.data(),&query,&lwork,&iquery,&liwork,&info);
  if (info) bad("L2 lift workspace failed");
  lwork=int(query); liwork=iquery;
  std::vector<double> work(lwork); std::vector<int> iwork(liwork);
  eigen(&no_vectors,&lower,&n,scratch.data(),&n,values.data(),work.data(),&lwork,iwork.data(),&liwork,&info);
  if (info) bad("L2 lift eigenvalues failed");
  const double largest=*std::max_element(values.begin(),values.end());
  const double tolerance=largest*double(lifted)*std::numeric_limits<double>::epsilon();
  const bool deficient=std::count_if(values.begin(),values.end(),[&](double v) { return v>tolerance; })<n;
  const double precision=deficient ? std::numeric_limits<float>::epsilon() : std::numeric_limits<double>::epsilon();
  auto factor=lower_cholesky(moment,n,precision*trace);
  if (factor.empty()) factor=lower_cholesky(moment,n,std::numeric_limits<float>::epsilon()*trace);
  if (factor.empty()) bad("L2 lift is not positive definite");
  std::vector<float> map(lifted*lifted);
  for (size_t i=0;i<lifted;++i) for (size_t j=0;j<lifted;++j) map[i*lifted+j]=float(factor[i+j*lifted]);
  return map;
}
// Inverse transpose of a lower-triangular row-major base map, so lifted queries times it are whitened.
inline std::vector<float> l2_query_map(const std::vector<float>& base_map,size_t lifted) {
  std::vector<double> inverse(lifted*lifted);
  for (size_t i=0;i<lifted;++i) for (size_t j=0;j<lifted;++j) inverse[i+j*lifted]=base_map[i*lifted+j];
  const char lower='L',general='N'; const int n=int(lifted); int info=0;
  dtrtri_(&lower,&general,&n,inverse.data(),&n,&info);
  if (info) bad("singular L2 lift");
  std::vector<float> map(lifted*lifted);
  for (size_t i=0;i<lifted;++i) for (size_t j=0;j<lifted;++j) map[i*lifted+j]=j>=i ? float(inverse[j+i*lifted]) : 0.0f;
  return map;
}
// Lifted input y = [x/s, last] (d + 1 wide), before the map: last is -|x/s|^2/2 for rows and 1 for queries.
inline void l2_lift_input(const float* x,size_t d,float scale,bool query,float* y) {
  double square=0;
  for (size_t j=0;j<d;++j) { y[j]=x[j]/scale; square+=double(y[j])*y[j]; }
  y[d]=query ? 1.0f : float(-square/2);
}
inline void normalize_row(const float* x,size_t d,float* out) {
  double square=0; for (size_t j=0;j<d;++j) square+=double(x[j])*x[j];
  const double norm=std::sqrt(square);
  for (size_t j=0;j<d;++j) out[j]=norm>0 ? float(x[j]/norm) : 0.0f;
}
} // namespace tq::tieredq_detail
