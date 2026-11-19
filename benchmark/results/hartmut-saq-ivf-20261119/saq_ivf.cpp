// SAQ (arxiv:2509.12086, upstream 2163ebc) IVF on the #55 arxiv partition, for faiss#5940.
//   build DATA CENTROIDS ASSIGNMENT BITS OUT: full-corpus PCA (fp64 covariance, all dimensions kept, as upstream's
//     pca.py), then SAQ's IVF on the given lists and row assignment (each row's list as uint32) at BITS per dimension.
//     Writes OUT/saq.index, OUT/pca.f32 (d x d, column per component) and OUT/mean.f32.
//   serve DATA OUT RESULT: loads the index in a fresh process; RAM is the RssAnon growth across load (after
//     malloc_trim). Then one thread on QUERY_CORE: per nprobe, single-query QPS (best of 3 passes over every query,
//     PCA of the query included), p50 and recall@k (id overlap with truth). Search is SAQ's own IVF::search (fp32 flat
//     coarse ranking, variance-pruned scan, default SearcherConfig), L2 distance.
// DATA holds meta.txt ("rows dimensions k f32"), base.f32, queries.f32, truth.i32.
#include <malloc.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <fmt/format.h>
#include "third/Eigen/Eigenvalues"
#include "index/ivf.hpp"

using namespace saqlib;
using Clock = std::chrono::steady_clock;
constexpr int build_threads = 32;
constexpr int timing_passes = 3;
const std::vector<size_t> nprobes = {8, 16, 32, 48, 64, 96, 128, 192, 256};

template <class T>
std::vector<T> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::vector<T> values(size_t(in.tellg()) / sizeof(T));
  in.seekg(0).read(reinterpret_cast<char*>(values.data()), std::streamsize(values.size() * sizeof(T)));
  return values;
}

template <class T>
void write_file(const std::string& path, const T* values, size_t count) {
  std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(values), std::streamsize(count * sizeof(T)));
}

struct Meta {
  size_t rows, d, k;
};

Meta read_meta(const std::string& data) {
  Meta meta{};
  std::string format;
  std::ifstream(data + "/meta.txt") >> meta.rows >> meta.d >> meta.k >> format;
  if (format != "f32") throw std::runtime_error("float32 data only");
  return meta;
}

double seconds_since(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

size_t padded_dimensions(size_t d) { return (d + kDimPaddingSize - 1) / kDimPaddingSize * kDimPaddingSize; }

// Full-dimension PCA basis, components in descending variance order.
std::pair<FloatRowMat, FloatVec> pca(const Eigen::Map<const FloatRowMat>& base) {
  const Eigen::RowVectorXd mean = base.cast<double>().colwise().mean();
  Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(base.cols(), base.cols());
  constexpr Eigen::Index chunk = 65536;
  for (Eigen::Index first = 0; first < base.rows(); first += chunk) {
    const Eigen::Index n = std::min(chunk, base.rows() - first);
    const Eigen::MatrixXd centered = base.middleRows(first, n).cast<double>().rowwise() - mean;
    covariance.noalias() += centered.transpose() * centered;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(covariance / double(base.rows()));
  return {solver.eigenvectors().rowwise().reverse().cast<float>(), mean.cast<float>()};
}

int build(const std::string& data, const std::string& centroids_path, const std::string& assignment_path, float bits,
          const std::string& out) {
  const Meta meta = read_meta(data);
  const auto base_values = read_file<float>(data + "/base.f32");
  const Eigen::Map<const FloatRowMat> base(base_values.data(), meta.rows, meta.d);
  const auto centroid_values = read_file<float>(centroids_path);
  const auto assignment = read_file<PID>(assignment_path);
  const size_t lists = centroid_values.size() / meta.d, padded = padded_dimensions(meta.d);
  if (assignment.size() != meta.rows) throw std::runtime_error("assignment does not cover every row");

  auto started = Clock::now();
  const auto [basis, mean] = pca(base);
  const double pca_s = seconds_since(started);
  write_file(out + "/pca.f32", basis.data(), basis.size());
  write_file(out + "/mean.f32", mean.data(), mean.size());

  FloatRowMat transformed = FloatRowMat::Zero(meta.rows, padded);
  transformed.leftCols(meta.d) = (base.rowwise() - mean) * basis;
  FloatRowMat centroids = FloatRowMat::Zero(lists, padded);
  centroids.leftCols(meta.d) =
      (Eigen::Map<const FloatRowMat>(centroid_values.data(), lists, meta.d).rowwise() - mean) * basis;

  QuantizeConfig config;
  config.avg_bits = bits;
  started = Clock::now();
  IVF ivf(meta.rows, padded, lists, config);
  ivf.construct(transformed, centroids, assignment.data(), build_threads);
  const double construct_s = seconds_since(started);
  ivf.save((out + "/saq.index").c_str());
  std::ofstream(out + "/build.json") << "{\"bits\": " << bits << ", \"lists\": " << lists << ", \"pca_s\": " << pca_s
                                     << ", \"construct_s\": " << construct_s << "}\n";
  return 0;
}

size_t rss_anon_bytes() {
  std::ifstream status("/proc/self/status");
  for (std::string line; std::getline(status, line);)
    if (line.rfind("RssAnon:", 0) == 0) return std::stoul(line.substr(8)) * 1024;
  return 0;
}

int serve(const std::string& data, const std::string& out, const std::string& result) {
  const Meta meta = read_meta(data);
  const auto queries = read_file<float>(data + "/queries.f32");
  const auto truth = read_file<int32_t>(data + "/truth.i32");
  const size_t query_count = queries.size() / meta.d, padded = padded_dimensions(meta.d);

  malloc_trim(0);
  const size_t rss_before = rss_anon_bytes();
  const auto basis_values = read_file<float>(out + "/pca.f32");
  const auto mean_values = read_file<float>(out + "/mean.f32");
  const Eigen::Map<const FloatRowMat> basis(basis_values.data(), meta.d, meta.d);
  const Eigen::Map<const FloatVec> mean(mean_values.data(), meta.d);
  IVF ivf;
  ivf.load((out + "/saq.index").c_str());
  malloc_trim(0);
  const size_t ram_bytes = rss_anon_bytes() - rss_before;

  cpu_set_t query_core;
  CPU_ZERO(&query_core);
  CPU_SET(std::atoi(std::getenv("QUERY_CORE")), &query_core);
  sched_setaffinity(0, sizeof(query_core), &query_core);
  Eigen::setNbThreads(1);

  SearcherConfig searcher;
  searcher.dist_type = DistType::L2Sqr;
  std::vector<PID> hits(query_count * meta.k);
  FloatVec projected = FloatVec::Zero(padded);
  std::ofstream report(result);
  report << "{\"kind\": \"saq\", \"ram_bytes\": " << ram_bytes << ", \"points\": [";
  for (size_t p = 0; p < nprobes.size(); ++p) {
    std::vector<double> latencies(query_count);
    double best_qps = 0;
    for (int pass = 0; pass < timing_passes; ++pass) {
      const auto pass_start = Clock::now();
      for (size_t q = 0; q < query_count; ++q) {
        const auto query_start = Clock::now();
        projected.head(meta.d) = (Eigen::Map<const FloatVec>(queries.data() + q * meta.d, meta.d) - mean) * basis;
        ivf.search<DistType::L2Sqr>(projected, meta.k, nprobes[p], searcher, hits.data() + q * meta.k);
        latencies[q] = seconds_since(query_start);
      }
      best_qps = std::max(best_qps, double(query_count) / seconds_since(pass_start));
    }
    size_t found = 0;
    for (size_t q = 0; q < query_count; ++q) {
      const int32_t* expected = truth.data() + q * meta.k;
      for (size_t i = 0; i < meta.k; ++i)
        found += std::find(expected, expected + meta.k, int32_t(hits[q * meta.k + i])) != expected + meta.k;
    }
    std::nth_element(latencies.begin(), latencies.begin() + query_count / 2, latencies.end());
    report << (p ? ", " : "") << "{\"nprobe\": " << nprobes[p] << ", \"qps\": " << best_qps
           << ", \"p50_us\": " << latencies[query_count / 2] * 1e6
           << ", \"recall\": " << double(found) / double(query_count * meta.k) << "}";
    std::cout << "nprobe " << nprobes[p] << " qps " << best_qps << " recall "
              << double(found) / double(query_count * meta.k) << std::endl;
  }
  report << "]}\n";
  return 0;
}

int main(int argc, char** argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);
  if (args.size() == 6 && args[0] == "build") return build(args[1], args[2], args[3], std::stof(args[4]), args[5]);
  if (args.size() == 4 && args[0] == "serve") return serve(args[1], args[2], args[3]);
  std::cerr << "usage: saq_ivf build DATA CENTROIDS ASSIGNMENT BITS OUT | serve DATA OUT RESULT\n";
  return 2;
}
