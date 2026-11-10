// Background /proc/self/status sampler. The kernel only keeps a total RSS high-water mark
// (VmHWM); this records peaks of anonymous (allocated) and file-backed (mapped input) RSS.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
namespace geometry_ann {
struct MemoryUsage { std::size_t rss = 0, anon = 0, file = 0, hwm = 0; };

inline MemoryUsage read_memory() {
    MemoryUsage usage;
    std::ifstream status("/proc/self/status"); std::string line;
    while (std::getline(status, line)) {
        std::istringstream fields(line); std::string key; std::size_t kb = 0;
        if (!(fields >> key >> kb)) continue;
        if (key == "VmRSS:") usage.rss = kb * 1024;
        else if (key == "RssAnon:") usage.anon = kb * 1024;
        else if (key == "RssFile:") usage.file = kb * 1024;
        else if (key == "VmHWM:") usage.hwm = kb * 1024;
    }
    return usage;
}

class MemorySampler {
public:
    explicit MemorySampler(std::chrono::milliseconds period = std::chrono::milliseconds(20))
        : worker_([this, period] { while (running_) { record(); std::this_thread::sleep_for(period); } }) {}
    MemorySampler(const MemorySampler&) = delete;
    MemorySampler& operator=(const MemorySampler&) = delete;
    ~MemorySampler() { stop(); }
    MemoryUsage stop() {
        if (running_.exchange(false)) worker_.join();
        record(); return peak_;
    }
private:
    void record() {
        auto now = read_memory();
        peak_.rss = std::max(peak_.rss, now.rss); peak_.anon = std::max(peak_.anon, now.anon);
        peak_.file = std::max(peak_.file, now.file); peak_.hwm = std::max(peak_.hwm, now.hwm);
    }
    std::atomic<bool> running_{true};
    MemoryUsage peak_;
    std::thread worker_;
};
}
