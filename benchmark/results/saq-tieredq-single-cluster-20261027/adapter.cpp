// Native SAQ single-cluster adapter: PCA preprocessing, production encode/scan, and owned storage.
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <numeric>
#include <vector>
#include <fmt/format.h>
#include "index/ivf.hpp"
using namespace saqlib;
constexpr size_t top_k = 10;

class ServingIVF : public IVF {
 public:
  using IVF::IVF;
  size_t owned_bytes() const {
    size_t bytes = sizeof(ServingIVF) + sizeof(FlatInitializer) + num_dim_ * num_cen_ * sizeof(float);
    bytes += parallel_clusters_.capacity() * sizeof(SaqCluData);
    for (const auto &cluster : parallel_clusters_) bytes += cluster.benchmark_owned_bytes();
    bytes += sizeof(SaqData) + saq_data_->data_variance.size() * sizeof(float);
    bytes += saq_data_->base_datas.capacity() * sizeof(BaseQuantizerData);
    bytes += saq_data_->quant_plan.capacity() * sizeof(SaqData::QuantPlanT::value_type);
    for (const auto &base : saq_data_->base_datas) {
      if (base.rotator) bytes += sizeof(utils::Rotator) + base.rotator->get_P().size() * sizeof(float);
    }
    return bytes;
  }
};

class Index {
 public:
  std::unique_ptr<ServingIVF> ivf;
  FloatRowMat pca;
  FloatVec mean;
  size_t dimensions;
  size_t padded;
  double encode_seconds;
  std::unique_ptr<BS::thread_pool<>> pool = std::make_unique<BS::thread_pool<>>(8);
  Index(const float *base, size_t count, size_t dims, const float *basis, const float *center,
        float bits, unsigned seed, const char *filename) : dimensions(dims), padded((dims+kDimPaddingSize-1)/kDimPaddingSize*kDimPaddingSize) {
    std::srand(seed + 1);
    pca = Eigen::Map<const FloatRowMat>(basis,dims,dims);
    mean = Eigen::Map<const FloatVec>(center,dims);
    FloatRowMat transformed = FloatRowMat::Zero(count,padded);
    transformed.leftCols(dims) = (Eigen::Map<const FloatRowMat>(base,count,dims).rowwise()-mean)*pca;
    FloatRowMat centroid = transformed.colwise().mean();
    std::vector<PID> assignments(count,0);
    QuantizeConfig config;
    config.avg_bits=bits;
    ivf=std::make_unique<ServingIVF>(count,padded,1,config);
    ivf->construct(transformed,centroid,assignments.data(),8);
    encode_seconds=ivf->benchmark_encode_seconds;
    ivf->save(filename);
    auto reloaded=std::make_unique<ServingIVF>();
    reloaded->load(filename);
    ivf=std::move(reloaded);
  }
  void one(const float *query, uint32_t *ids, bool fullscan) {
    FloatVec projected=FloatVec::Zero(padded);
    projected.head(dimensions)=(Eigen::Map<const FloatVec>(query,dimensions)-mean)*pca;
    SearcherConfig config;
    config.dist_type=DistType::L2Sqr;
    if (!fullscan) {
      ivf->search<DistType::L2Sqr>(projected,top_k,1,config,ids);
      return;
    }
    SaqCluEstimator<DistType::L2Sqr> estimator(*ivf->get_saq_data(),config,projected);
    utils::ResultPool nearest(top_k,false);
    for (const auto &cluster : ivf->get_pclusters()) {
      estimator.prepare(&cluster);
      alignas(64) __m512 fast_distances[KFastScanSize / 16];
      for (size_t i=0;i<cluster.num_vec_;++i) {
        if (i % KFastScanSize == 0) estimator.compFastDist(i / KFastScanSize,fast_distances);
        nearest.insert(cluster.ids()[i],estimator.compAccurateDist(i));
      }
    }
    nearest.copy_results(ids);
  }
  void search(const float *queries, size_t count, uint32_t *ids, size_t threads, bool fullscan) {
    if (threads==1 || count==1) {
      for (size_t i=0;i<count;++i) one(queries+i*dimensions,ids+i*top_k,fullscan);
      return;
    }
    pool->detach_loop(size_t(0),count,[&](size_t i){one(queries+i*dimensions,ids+i*top_k,fullscan);});
    pool->wait();
  }
  size_t owned_bytes() const {
    return sizeof(Index)+ivf->owned_bytes()+(pca.size()+mean.size())*sizeof(float);
  }
};

extern "C" {
void *saq_build(const float *base, size_t count, size_t dims, const float *basis,
                const float *mean, float bits, unsigned seed, const char *filename) {
  return new Index(base,count,dims,basis,mean,bits,seed,filename);
}
void saq_search(void *ptr,const float *queries,size_t count,uint32_t *ids,size_t threads, bool fullscan) {
  static_cast<Index *>(ptr)->search(queries,count,ids,threads,fullscan);
}
void saq_stats(void *ptr,double *values) {
  const auto &x=*static_cast<Index *>(ptr);
  values[0]=x.owned_bytes();values[1]=x.encode_seconds;
}
void saq_free(void *ptr) { delete static_cast<Index *>(ptr); }
}
