// TieredQFastBlocks: exact pack/unpack round trips and in-place scores equal to the fast index over the same rows,
// for every format a model admits, at counts that end on short tails.
#include "tieredq/tieredq.h"
#include "tieredq/tieredq_blocks.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }

constexpr uint32_t dimensions = 64, centroids = 8;
constexpr size_t training_rows = 512;
constexpr size_t counts[] = {1, 17, 100, 256, 300, 513};
// Candidate formats: levels (x86 / ARM ranges), packed code words and nibble lookups (see tieredq_blocks.cpp).
constexpr uint32_t candidate_formats[] = {63u << 8 | 1, 127u << 8 | 1, 0u << 8 | 1, 2u};

std::vector<float> normal_rows(size_t rows, uint32_t seed) {
  std::mt19937 random(seed); std::normal_distribution<float> normal;
  std::vector<float> data(rows * dimensions);
  for (auto& value : data) value = normal(random);
  return data;
}

// Scores of documents 0..count by the fast index (search_fast over all of them, mapped back by id).
std::vector<float> index_scores(const dv::TieredQ& codec, const float* rows, size_t count, const float* query) {
  dv::TieredQIndex index(codec);
  index.add(rows, count);
  std::vector<float> scores(count);
  for (const auto& hit : index.search_fast(query, count)) scores[hit.id] = hit.score;
  return scores;
}

void check_format(const dv::TieredQ& codec, const dv::TieredQFastBlocks& blocks, const std::vector<float>& data,
                  const std::vector<float>& query, bool native) {
  const size_t stride = codec.bytes_per_vector();
  for (size_t count : counts) {
    const auto rows = codec.encode(data.data(), count);
    // Odd offset inside a larger buffer: blocks need no particular alignment.
    std::vector<uint8_t> buffer(blocks.block_bytes(count) + 1, 0xA5);
    uint8_t* block = buffer.data() + 1;
    blocks.pack(rows.data(), count, block);
    std::vector<uint8_t> unpacked(count * stride);
    blocks.unpack(block, count, unpacked.data());
    require(unpacked == rows, "fast block unpack differs from the packed rows");
    auto scanner = blocks.scanner(query.data());
    std::vector<float> scores(count);
    scanner.score(block, count, scores.data());
    if (native) {
      const auto expected = index_scores(codec, data.data(), count, query.data());
      for (size_t doc = 0; doc < count; ++doc)
        require(std::fabs(scores[doc] - expected[doc]) <= 1e-5f * (1 + std::fabs(expected[doc])), "block scores differ from the fast index");
    } else {
      std::vector<float> exact(count);
      codec.score(rows.data(), count, codec.query(query.data()), exact.data());
      double error = 0, scale = 0;
      for (size_t doc = 0; doc < count; ++doc) { error += std::fabs(scores[doc] - exact[doc]); scale += std::fabs(exact[doc]); }
      require(error <= 0.05 * scale + 1e-3 * double(count), "block scores drift from exact codec scores");
    }
  }
}
}  // namespace

void tieredq_blocks_tests() {
  const auto training = normal_rows(training_rows, 3), data = normal_rows(counts[std::size(counts) - 1], 5), query = normal_rows(1, 7);
  for (uint32_t bits_per_dim : {1u, 2u, 4u}) {
    auto config = dv::TieredQConfig::target_code_bits(dimensions, dimensions * bits_per_dim);
    config.num_centroids = centroids;
    const auto codec = dv::TieredQ::train(config, training.data(), training_rows);
    const dv::TieredQFastBlocks native(codec);
    check_format(codec, native, data, query, true);
    const dv::TieredQFastBlocks reopened(codec, native.format());
    require(reopened.format() == native.format() && reopened.block_bytes(300) == native.block_bytes(300), "persisted format reopens differently");
    for (uint32_t format : candidate_formats) {
      if (format == native.format()) continue;
      try { check_format(codec, dv::TieredQFastBlocks(codec, format), data, query, false); }
      catch (const std::runtime_error& e) { if (!std::string(e.what()).ends_with("does not fit this model") && !std::string(e.what()).ends_with("identify this model's codes")) throw; }
    }
  }
  bool rejected = false;
  try { dv::TieredQFastBlocks(dv::TieredQ::train({dimensions, dimensions, centroids}, training.data(), training_rows), 0); }
  catch (const std::runtime_error&) { rejected = true; }
  require(rejected, "unknown fast block format accepted");
}
