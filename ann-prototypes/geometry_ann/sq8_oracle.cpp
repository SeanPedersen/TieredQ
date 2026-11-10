// Batched exhaustive diagnostic: one sequential corpus pass, independent of routing.
#include "benchmark_options.h"
#include "dataset.h"
#include "internal.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <queue>
#include <stdexcept>
using namespace geometry_ann;
int main(int argc,char** argv) {
 try {
    auto config=parse_options(argc,argv);
    auto data=load_dataset(config.base,config.queries,config.truth,config.query_count);
    auto encoded=config.int8_base.empty()?QuantizedDataset{}:load_quantized(config.int8_base,config.int8_scales,data.count(),data.dimension,config.int8_offsets);
    if (!encoded.codes.empty()) data.release_base();
    for (float value:data.queries) if (!std::isfinite(value)) throw std::runtime_error("nonfinite query");
    std::vector<std::priority_queue<Neighbor>> best(data.truth.size());
    auto start=std::chrono::steady_clock::now();
    for (std::size_t id=0;id<data.count();++id) {
        std::span<const float> vector;
        std::span<const std::int8_t> code;
        if (!encoded.codes.empty()) code=encoded.codes.subspan(id*data.dimension,data.dimension);
        else {
            vector=data.base.subspan(id*data.dimension,data.dimension);
            for (float value:vector) if (!std::isfinite(value)) throw std::runtime_error("nonfinite base");
        }
        for (std::size_t q=0;q<data.truth.size();++q) {
            auto query=std::span(data.queries).subspan(q*data.dimension,data.dimension);
            auto& heap=best[q];double distance=0;
            if (!encoded.codes.empty()) {
                if (heap.size()==10 && sq8_lower_bound(query,code,encoded.scales,encoded.offsets)>heap.top().distance) continue;
                distance=sq8_l2(query,code,encoded.scales,encoded.offsets);
            } else {
                if (heap.size()==10 && scan_lower_bound(query,vector)>heap.top().distance) continue;
                distance=l2(query,vector);
            }
            Neighbor candidate{distance,Id(id)};
            if (heap.size()<10) heap.push(candidate);
            else if (candidate<heap.top()) { heap.pop();heap.push(candidate); }
        }
    }
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::size_t hits=0;
    for (std::size_t q=0;q<data.truth.size();++q) while (!best[q].empty()) {
        auto n=best[q].top();best[q].pop();hits+=std::find(data.truth[q].begin(),data.truth[q].end(),n.id)!=data.truth[q].end();
    }
    std::cout << "n,dimension,nq,recall10,elapsed_s,codes_bytes,vector_storage\n" << data.count() << ',' << data.dimension << ','
              << data.truth.size() << ',' << double(hits)/(10*data.truth.size()) << ',' << seconds << ','
              << encoded.codes.size() << ',' << (encoded.codes.empty()?"fp32":"int8_sq_per_dim") << '\n';
 } catch (const std::exception& error) { std::cerr << error.what() << '\n';return 1; }
}
