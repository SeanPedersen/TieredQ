// Peak resident set during a measured section: resets VmHWM (clear_refs 5) at construction, reports the peak and
// the growth over the resident set at the start (the loaded data).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace repro {

inline long StatusKiB(const char* field) {
    std::ifstream status("/proc/self/status");
    std::string line;
    const std::string prefix = std::string(field) + ":";
    while (std::getline(status, line)) {
        if (line.rfind(prefix, 0) == 0) return std::stol(line.substr(prefix.size()));
    }
    return -1;
}

inline unsigned Threads() {
    const char* threads = std::getenv("THREADS");
    return threads ? unsigned(std::stoul(threads)) : 0u;
}

class PeakRss {
  public:
    PeakRss() {
        std::ofstream("/proc/self/clear_refs") << "5";
        start_kib = StatusKiB("VmRSS");
    }
    void Print() const {
        const long peak_kib = StatusKiB("VmHWM");
        printf("Peak RSS: %.2f GB | training overhead: %.2f GB\n", peak_kib / 1048576.0,
               (peak_kib - start_kib) / 1048576.0);
    }

  private:
    long start_kib;
};

}  // namespace repro
