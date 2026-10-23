// Reproducible single-thread packed-scan benchmark; synthetic data isolates backend costs.
#include <tieredq/tieredq.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <numeric>
#include <random>

namespace {
using Clock=std::chrono::steady_clock;
constexpr size_t dimensions=384, fitting_rows=1024, database_rows=100003;
constexpr size_t query_count=32, rounds=3, neighbors=10;
double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
std::vector<uint32_t> top(const std::vector<float>& scores) {
  std::vector<uint32_t> ids(scores.size()); std::iota(ids.begin(),ids.end(),0);
  std::partial_sort(ids.begin(),ids.begin()+neighbors,ids.end(),[&](auto a,auto b) {
    return scores[a]>scores[b] || (scores[a]==scores[b] && a<b);
  });
  ids.resize(neighbors); return ids;
}
double percentile(std::vector<double> values,double quantile) {
  std::sort(values.begin(),values.end()); return values[size_t(quantile*(values.size()-1))];
}
}
int main(int argc,char** argv) {
  if (argc>2) throw std::runtime_error("usage: tieredq_scan_benchmark [scale_embeddings.f32]");
  std::mt19937 rng(42); std::normal_distribution<float> normal;
  std::vector<float> fitting(fitting_rows*dimensions),queries(query_count*dimensions);
  for (auto* values:{&fitting,&queries}) for (auto& value:*values) value=normal(rng)/std::sqrt(float(dimensions));
  std::vector<float> database;
  size_t count=database_rows;
  if (argc==2) {
    constexpr size_t real_rows=100000, real_fit=8000, real_queries=2000, calibration_rows=10000;
    std::ifstream input(argv[1],std::ios::binary);
    const size_t total_rows=real_rows+real_fit+calibration_rows+real_queries;
    std::vector<float> embeddings(total_rows*dimensions);
    if (!input.read(reinterpret_cast<char*>(embeddings.data()),embeddings.size()*sizeof(float)))
      throw std::runtime_error("expected raw float32 scale embeddings: 120000 x 384");
    database.assign(embeddings.begin(),embeddings.begin()+real_rows*dimensions);
    fitting.assign(embeddings.begin()+real_rows*dimensions,embeddings.begin()+(real_rows+real_fit)*dimensions);
    for (size_t i=0;i<query_count;++i)
      std::copy_n(embeddings.data()+(real_rows+real_fit+calibration_rows+i*(real_queries-1)/(query_count-1))*dimensions,
                  dimensions,queries.data()+i*dimensions);
    count=real_rows;
  }
  const size_t training_count=fitting.size()/dimensions;
  for (uint32_t bits:{808u,1176u,1560u}) {
    std::cout<<"training bits="<<bits<<std::endl;
    auto codec=dv::TieredQ::train({dimensions,bits,256,true},fitting.data(),training_count,42);
    // Synthetic runs repeat fitting rows; supplied embeddings use the complete database.
    const auto encoded=codec.encode(database.empty() ? fitting.data() : database.data(),database.empty() ? training_count : count);
    const size_t stride=codec.bytes_per_vector();
    std::vector<uint8_t> rows(count*stride),planes((count+15)/16*16*stride);
    for (size_t i=0;i<count;++i) std::copy_n(encoded.data()+(i%(database.empty() ? training_count : count))*stride,stride,rows.data()+i*stride);
    codec.pack(rows.data(),count,planes.data());
    std::vector<float> reference(count),scores(count);
    size_t agreements=0,scalar_overlap=0,automatic_overlap=0; float max_error=0;
    std::vector<float> exact(count);

    for (size_t i=0;i<query_count;++i) {
      auto q=codec.query(queries.data()+i*dimensions);
      codec.score_planes(planes.data(),count,q,reference.data(),dv::TieredQKernel::scalar,false);
      codec.score_planes(planes.data(),count,q,scores.data(),dv::TieredQKernel::automatic,false);
      const auto scalar_hits=top(reference),automatic_hits=top(scores);
      agreements+=scalar_hits==automatic_hits;
      if (!database.empty()) {
        for (size_t row=0;row<count;++row) {
          float dot=0;
          for (size_t col=0;col<dimensions;++col) dot+=database[row*dimensions+col]*queries[i*dimensions+col];
          exact[row]=dot;
        }
        for (auto id:top(exact)) {
          scalar_overlap+=std::find(scalar_hits.begin(),scalar_hits.end(),id)!=scalar_hits.end();
          automatic_overlap+=std::find(automatic_hits.begin(),automatic_hits.end(),id)!=automatic_hits.end();
        }
      }
      for (size_t j=0;j<count;++j) max_error=std::max(max_error,std::fabs(scores[j]-reference[j]));
    }
    std::cout<<"model_bytes="<<codec.model_bytes()<<" payload_bytes="<<rows.size()
             <<" top10_identical="<<agreements<<"/"<<query_count<<" scalar_recall="<<double(scalar_overlap)/(query_count*neighbors)
             <<" automatic_recall="<<double(automatic_overlap)/(query_count*neighbors)<<" max_score_error="<<max_error<<'\n';
    std::vector<double> preparation[2],scan[2],selection[2],total[2];
    uint64_t checksum=0;
    for (size_t round=0;round<=rounds;++round) for (size_t i=0;i<query_count;++i) for (size_t turn=0;turn<2;++turn) {
      const size_t mode=(turn+round+i)%2;
      auto start=Clock::now(); auto q=codec.query(queries.data()+i*dimensions); const double prep=milliseconds(start);
      start=Clock::now();
      codec.score_planes(planes.data(),count,q,scores.data(),mode ? dv::TieredQKernel::automatic : dv::TieredQKernel::scalar,false);
      const double scoring=milliseconds(start);
      start=Clock::now(); const auto hits=top(scores); const double select=milliseconds(start); checksum+=hits.front();
      if (!round) continue;
      preparation[mode].push_back(prep); scan[mode].push_back(scoring); selection[mode].push_back(select); total[mode].push_back(prep+scoring+select);
    }
    for (size_t mode=0;mode<2;++mode) std::cout<<(mode ? "automatic" : "scalar")
      <<" prep_p50_ms="<<percentile(preparation[mode],0.5)<<" scan_p50_ms="<<percentile(scan[mode],0.5)
      <<" scan_p95_ms="<<percentile(scan[mode],0.95)<<" topk_p50_ms="<<percentile(selection[mode],0.5)
      <<" total_p50_ms="<<percentile(total[mode],0.5)<<" total_p95_ms="<<percentile(total[mode],0.95)<<" checksum="<<checksum<<std::endl;
  }
}
