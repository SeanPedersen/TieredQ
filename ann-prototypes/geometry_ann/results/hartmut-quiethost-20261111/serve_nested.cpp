// C++ counterpart of serve.py (#52 quiet-host rerun): loads the same saved nested index, attaches its tier file with
// direct reads (32 in flight, the Python default), and measures held-out recall@10 and per-query latency at the
// index's default candidates, one query at a time. Run alone on one pinned core with one OpenMP thread. Writes the
// same CSV columns as serve.py.
// Usage: serve_nested INDEX.tq TIER.rows QUERIES.fvecs TRUTH.ivecs COUNT NPROBES(comma list) OUT.csv
#include "tieredq/tieredq_nested.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t top_k = 10, direct_queue_depth = 32;

template<class T> std::vector<T> read_vecs(const std::string& path, size_t count, size_t keep) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  int32_t width = 0;
  std::vector<T> out;
  for (size_t row = 0; row < count && in.read(reinterpret_cast<char*>(&width), sizeof(width)); ++row) {
    std::vector<T> values(static_cast<size_t>(width));
    in.read(reinterpret_cast<char*>(values.data()), std::streamsize(values.size() * sizeof(T)));
    out.insert(out.end(), values.begin(), values.begin() + std::min(keep, values.size()));
  }
  return out;
}

size_t anonymous_rss() {
  std::ifstream status("/proc/self/status");
  for (std::string line; std::getline(status, line);)
    if (line.rfind("RssAnon:", 0) == 0) return std::stoul(line.substr(8)) * 1024;
  return 0;
}

double percentile(std::vector<double> values, double share) {
  std::sort(values.begin(), values.end());
  const double position = share * double(values.size() - 1);
  const size_t below = size_t(position);
  const double above = below + 1 < values.size() ? values[below + 1] : values[below];
  return values[below] + (position - double(below)) * (above - values[below]);  // numpy's linear interpolation
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) { std::fprintf(stderr, "usage: serve_nested INDEX TIER QUERIES TRUTH COUNT NPROBES OUT\n"); return 2; }
  std::ifstream in(argv[1], std::ios::binary);
  const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  auto index = dv::TieredQNestedIVFIndex::deserialize(bytes);
  index.attach_tier(std::make_shared<dv::TieredQFileRows>(argv[2], dv::TieredQFileReads::direct, direct_queue_depth));
  const size_t count = std::stoul(argv[5]), d = index.index().codec().config().dimensions;
  const auto queries = read_vecs<float>(argv[3], count, d);
  const auto truth = read_vecs<int32_t>(argv[4], count, top_k);
  std::ofstream report(argv[7]);
  report << "nprobe,recall10,p50_us,p99_us,mean_us,anon_bytes\n";
  std::stringstream nprobes(argv[6]);
  for (std::string field; std::getline(nprobes, field, ',');) {
    const size_t nprobe = std::stoul(field);
    index.search(queries.data(), top_k, nprobe);  // warm, as serve.py
    std::vector<double> micros(count);
    size_t hits = 0;
    for (size_t q = 0; q < count; ++q) {
      const auto started = std::chrono::steady_clock::now();
      const auto found = index.search(queries.data() + q * d, top_k, nprobe);
      micros[q] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
      for (const auto& hit : found) hits += std::count(truth.begin() + q * top_k, truth.begin() + (q + 1) * top_k, int32_t(hit.id));
    }
    double mean = 0;
    for (double value : micros) mean += value / double(count);
    char line[160];
    std::snprintf(line, sizeof(line), "%zu,%.4f,%.1f,%.1f,%.1f,%zu\n", nprobe, double(hits) / double(count * top_k),
                  percentile(micros, 0.5), percentile(micros, 0.99), mean, anonymous_rss());
    report << line << std::flush;
  }
}
