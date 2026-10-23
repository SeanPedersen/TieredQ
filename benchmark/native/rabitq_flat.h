// Flat RaBitQ storage and orchestration over official native codec kernels; IDs are row positions.
#pragma once
#include <rabitqlib/index/estimator.hpp>
#include <rabitqlib/quantization/rabitq.hpp>
#include <rabitqlib/utils/buffer.hpp>
#include <rabitqlib/utils/rotator.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace benchmark {
class AlignedBuffer {
  static constexpr size_t alignment = 64;
  struct alignas(alignment) Block { std::array<char, alignment> bytes{}; };
  std::vector<Block> blocks_;
 public:
  explicit AlignedBuffer(size_t bytes) : blocks_((bytes+alignment-1)/alignment) {}
  char* data() { return reinterpret_cast<char*>(blocks_.data()); }
  const char* data() const { return reinterpret_cast<const char*>(blocks_.data()); }
  size_t bytes() const { return blocks_.size()*sizeof(Block); }
  size_t allocated_bytes() const { return blocks_.capacity()*sizeof(Block); }
};

class RaBitQFlat {
  using Clock = std::chrono::steady_clock;
  static constexpr size_t batch_size = rabitqlib::fastscan::kBatchSize;
  static constexpr size_t rotation_alignment = 64;
  static constexpr std::array<char,8> magic = {'R','Q','F','L','A','T','0','1'};
  static constexpr size_t header_fields = 5;
  static constexpr size_t header_bytes = magic.size()+header_fields*sizeof(uint64_t);
  size_t count_, dimensions_, padded_, extra_bits_;
  rabitqlib::MetricType metric_;
  std::unique_ptr<rabitqlib::Rotator<float>> rotator_;
  std::vector<float> center_;
  AlignedBuffer binary_, extra_;
  double encode_s_ = 0;
  float (*extra_ip_)(const float*,const uint8_t*,size_t);

  size_t binary_stride() const { return rabitqlib::BatchDataMap<float>::data_bytes(padded_); }
  size_t extra_stride() const { return rabitqlib::ExDataMap<float>::data_bytes(padded_,extra_bits_); }

 public:
  static constexpr bool high_accuracy = true;
  RaBitQFlat(const float* base,size_t count,size_t dimensions,size_t bits,bool l2)
      : count_(count), dimensions_(dimensions),
        padded_(rabitqlib::round_up_to_multiple(dimensions,rotation_alignment)), extra_bits_(bits-1),
        metric_(l2 ? rabitqlib::METRIC_L2 : rabitqlib::METRIC_IP),
        rotator_(rabitqlib::choose_rotator<float>(dimensions,rabitqlib::RotatorType::FhtKacRotator,padded_)),
        center_(padded_), binary_(((count+batch_size-1)/batch_size)*binary_stride()),
        extra_(count*extra_stride()), extra_ip_(rabitqlib::select_excode_ipfunc(extra_bits_)) {
    if (rotator_->size()!=padded_) throw std::runtime_error("Unexpected native rotation padding");
    std::vector<float> mean(dimensions_,0);
    for (size_t row=0;row<count_;++row) for (size_t col=0;col<dimensions_;++col)
      mean[col]+=base[row*dimensions_+col]/count_;
    rotator_->rotate(mean.data(),center_.data());
    const auto start=Clock::now();
    const size_t blocks=(count_+batch_size-1)/batch_size;
    #pragma omp parallel for schedule(static)
    for (size_t block=0;block<blocks;++block) {
      const size_t first=block*batch_size, rows=std::min(batch_size,count_-first);
      std::vector<float> rotated(rows*padded_);
      for (size_t row=0;row<rows;++row)
        rotator_->rotate(base+(first+row)*dimensions_,rotated.data()+row*padded_);
      rabitqlib::quant::quantize_split_batch(rotated.data(),center_.data(),rows,padded_,extra_bits_,
          binary_.data()+block*binary_stride(), extra_stride() ? extra_.data()+first*extra_stride() : nullptr, metric_);
    }
    encode_s_=std::chrono::duration<double>(Clock::now()-start).count();
  }
  double encode_seconds() const { return encode_s_; }
  void search(const float* vector,size_t k,uint32_t* ids) const {
    std::vector<float> rotated(padded_);
    rotator_->rotate(vector,rotated.data());
    const float distance=std::sqrt(rabitqlib::euclidean_sqr(rotated.data(),center_.data(),padded_));
    rabitqlib::buffer::SearchBuffer<float> top(k);
    rabitqlib::SplitBatchQuery<float> query(rotated.data(),padded_,extra_bits_,metric_,high_accuracy);
    const float center_ip=metric_==rabitqlib::METRIC_IP
        ? rabitqlib::dot_product<float>(rotated.data(),center_.data(),padded_) : 0;
    query.set_g_add(distance,center_ip);
    for (size_t first=0;first<count_;first+=batch_size) {
      const size_t block=first/batch_size, rows=std::min(batch_size,count_-first);
      std::array<float,batch_size> estimates,lower,first_bit_ip;
      rabitqlib::split_batch_estdist(binary_.data()+block*binary_stride(),query,padded_,
          estimates.data(),lower.data(),first_bit_ip.data(),high_accuracy);
      float cutoff=top.top_dist();
      for (size_t row=0;row<rows;++row) {
        if (!extra_bits_) {
          top.insert(static_cast<uint32_t>(first+row),estimates[row]);
          continue;
        }
        if (BENCH_FULL_SCORE || lower[row]<cutoff) {
          const float score=rabitqlib::split_distance_boosting(extra_.data()+(first+row)*extra_stride(),
              extra_ip_,query,padded_,extra_bits_,first_bit_ip[row]);
          top.insert(static_cast<uint32_t>(first+row),score);
          cutoff=top.top_dist();
        }
      }
    }
    top.copy_results(ids);
  }
  size_t storage_bytes() const {
    return header_bytes+rotator_->dump_bytes()+center_.capacity()*sizeof(float)
        +binary_.allocated_bytes()+extra_.allocated_bytes();
  }
  void save(const std::string& path) const {
    std::ofstream out(path,std::ios::binary);
    const std::array<uint64_t,header_fields> fields={count_,dimensions_,padded_,extra_bits_,uint64_t(metric_)};
    out.write(magic.data(),magic.size());
    out.write(reinterpret_cast<const char*>(fields.data()),sizeof(fields));
    rotator_->save(out);
    out.write(reinterpret_cast<const char*>(center_.data()),center_.size()*sizeof(float));
    out.write(binary_.data(),binary_.bytes());
    if (extra_.bytes()) out.write(extra_.data(),extra_.bytes());
    if (!out) throw std::runtime_error("Cannot save flat RaBitQ storage");
  }
};
} // namespace benchmark
