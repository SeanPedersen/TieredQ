// #55 gate: single-thread scan time of the shared fast-scan kernels over one whole-run TieredQFastBlocks block (both
// sides), and over the same rows as 256-document tiles (B only, TIEREDQ_TILES). Synthetic Gaussian rows, so A and B
// scan identical bytes; per query the scanner is prepared outside the timed region.
// Training and encoding use every allowed core; scans then run alone on QUERY_CORE.
// Usage: QUERY_CORE=2 kernel_bench DIMENSIONS BITS_PER_DIM_X100 ROWS QUERIES  -> one JSON line
#include <tieredq/tieredq.h>
#include <tieredq/tieredq_blocks.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <sched.h>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr size_t training_rows = 65536, passes = 5;

std::vector<float> gaussian(size_t rows, size_t dimensions, uint32_t seed) {
  std::mt19937 random(seed); std::normal_distribution<float> normal;
  std::vector<float> data(rows * dimensions);
  for (auto& value : data) value = normal(random);
  return data;
}

template<class Scan> double best_ns_per_row(size_t queries, size_t rows, Scan scan) {
  double best = 1e300;
  for (size_t pass = 0; pass < passes; ++pass) {
    double total = 0;
    for (size_t q = 0; q < queries; ++q) total += scan(q);
    best = std::min(best, total);
  }
  return best / double(queries * rows);
}
}  // namespace

int main(int argc, char** argv) {
  const size_t dimensions = std::stoul(argv[1]), rows = std::stoul(argv[3]), queries = std::stoul(argv[4]);
  const uint32_t code_bits = uint32_t(dimensions * std::stoul(argv[2]) / 100);
  auto config = tq::TieredQConfig::target_code_bits(uint32_t(dimensions), code_bits);
  config.max_scan_bits = tq::TieredQIVFIndex::automatic_scan_bits(config);
  const auto training = gaussian(training_rows, dimensions, 1);
  const auto codec = tq::TieredQ::train(config, training.data(), training_rows);
  const tq::TieredQFastBlocks blocks(codec);
  const auto data = gaussian(rows, dimensions, 2), query_rows = gaussian(queries, dimensions, 3);
  const auto codes = codec.encode(data.data(), rows);
  std::vector<uint8_t> block(blocks.block_bytes(rows));
  blocks.pack(codes.data(), rows, block.data());
  cpu_set_t query_core; CPU_ZERO(&query_core); CPU_SET(std::atoi(std::getenv("QUERY_CORE")), &query_core);
  if (sched_setaffinity(0, sizeof(query_core), &query_core)) return 1;
  std::vector<tq::TieredQFastBlocks::Scanner> scanners;
  for (size_t q = 0; q < queries; ++q) scanners.push_back(blocks.scanner(query_rows.data() + q * dimensions));
  std::vector<float> scores(rows);
  const double block_ns = best_ns_per_row(queries, rows, [&](size_t q) {
    const auto started = Clock::now();
    scanners[q].score(block.data(), rows, scores.data());
    return std::chrono::duration<double, std::nano>(Clock::now() - started).count();
  });
  std::printf("{\"dimensions\": %zu, \"code_bits\": %u, \"format\": %u, \"rows\": %zu, \"queries\": %zu, \"block_ns_per_row\": %.4f",
              dimensions, code_bits, blocks.format(), rows, queries, block_ns);
#ifdef TIEREDQ_TILES
  constexpr size_t slots = tq::TieredQFastBlocks::tile_slots;
  const size_t tiles = (rows + slots - 1) / slots, tile_bytes = blocks.tile_bytes();
  std::vector<uint8_t> tiled(tiles * tile_bytes, 0);
  for (size_t row = 0; row < rows; ++row)
    blocks.write_row(tiled.data() + row / slots * tile_bytes, row % slots, codes.data() + row * codec.bytes_per_vector());
  const double tile_ns = best_ns_per_row(queries, rows, [&](size_t q) {
    const auto started = Clock::now();
    for (size_t t = 0; t < tiles; ++t)
      scanners[q].score_tile(tiled.data() + t * tile_bytes, std::min(slots, rows - t * slots), scores.data() + t * slots);
    return std::chrono::duration<double, std::nano>(Clock::now() - started).count();
  });
  std::printf(", \"tile_ns_per_row\": %.4f, \"tile_bytes\": %zu, \"block_bytes\": %zu", tile_ns, tile_bytes, block.size());
#endif
  std::printf("}\n");
}
