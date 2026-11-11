// Opt-in training/encoding timing and binary diagnostics. Profiling is silent and clock-free when disabled.
#pragma once
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace tq::tieredq_detail {
template<class Value> void dump_training_stage(const char* stage,const std::vector<Value>& values) {
  const char* prefix=std::getenv("TQ_TRAIN_DUMP");
  if (!prefix) return;
  const std::string path=std::string(prefix)+"."+stage;
  std::ofstream out(path,std::ios::binary);
  out.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(Value));
  if (!out) throw std::runtime_error("TieredQ: cannot write training diagnostic "+path);
}

class StageProfile {
 public:
  using Clock = std::chrono::steady_clock;
  explicit StageProfile(const char* label="tieredq_train_profile", StageProfile* sink=nullptr) : label_(label), sink_(sink), enabled_(std::getenv("TQ_PROFILE") != nullptr), last_(enabled_ ? Clock::now() : Clock::time_point{}) {}
  StageProfile(const StageProfile&) = delete;
  StageProfile& operator=(const StageProfile&) = delete;
  // Charges the time since the previous mark to `stage`; repeated names accumulate.
  void mark(const char* stage) {
    if (!enabled_) return;
    const auto now = Clock::now();
    const double elapsed = std::chrono::duration<double>(now - last_).count();
    last_ = now;
    for (auto& [name, seconds] : stages_) if (name == stage) { seconds += elapsed; return; }
    stages_.emplace_back(stage, elapsed);
  }
  void reset() { if (enabled_) last_=Clock::now(); }
  ~StageProfile() {
    if (!enabled_) return;
    if (sink_) {
      std::lock_guard<std::mutex> lock(sink_->mutex_);
      for (const auto& [name, seconds]:stages_) {
        bool found=false;
        for (auto& [target, value]:sink_->stages_) if (target==name) { value+=seconds; found=true; break; }
        if (!found) sink_->stages_.emplace_back(name,seconds);
      }
      return;
    }
    std::string line = "{\"" + std::string(label_) + "\":{";
    for (size_t i = 0; i < stages_.size(); ++i)
      line += (i ? ",\"" : "\"") + stages_[i].first + "\":" + std::to_string(stages_[i].second);
    std::cerr << line << "}}\n";
  }

 private:
  const char* label_;
  StageProfile* sink_;
  std::mutex mutex_;
  bool enabled_;
  Clock::time_point last_;
  std::vector<std::pair<std::string, double>> stages_;
};
}  // namespace tq::tieredq_detail
