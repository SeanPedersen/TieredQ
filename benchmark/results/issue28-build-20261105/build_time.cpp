// One-shot paired build timing on the same cached OpenAI rows and seeded training sample.
#include <tieredq/tieredq.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <algorithm>
int main() {
 constexpr size_t n=100000,d=1536,sample_rows=32768;
 std::vector<float> rows(n*d);
 std::ifstream in("/home/sean/experiments/TieredQ-issue20-ship-20261029/data/openai/base.f32",std::ios::binary);
 if(!in.read(reinterpret_cast<char*>(rows.data()),rows.size()*sizeof(float))) return 1;
 std::vector<uint32_t> ids(n);std::iota(ids.begin(),ids.end(),0);std::mt19937_64 rng(0);std::shuffle(ids.begin(),ids.end(),rng);ids.resize(sample_rows);
 auto config=dv::TieredQConfig::target_code_bits(d,4*d);config.metric=dv::TieredQMetric::cosine;
 auto start=std::chrono::steady_clock::now();
 auto codec=dv::TieredQ::train_sampled(config,rows.data(),n,ids.data(),ids.size(),0);
 auto trained=std::chrono::steady_clock::now();
 const auto model=codec.model_id();
 dv::TieredQIndex index(std::move(codec));index.add(rows.data(),n);
 auto done=std::chrono::steady_clock::now();
 std::cout<<"{\"train_s\":"<<std::chrono::duration<double>(trained-start).count()<<",\"add_s\":"<<std::chrono::duration<double>(done-trained).count()<<",\"build_s\":"<<std::chrono::duration<double>(done-start).count()<<",\"model_id\":"<<model<<",\"storage_bytes\":"<<index.storage_bytes()<<"}"<<std::endl;
}
