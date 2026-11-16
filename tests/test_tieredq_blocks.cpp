// TieredQFastBlocks: exact pack/unpack round trips and in-place scores equal to the fast index over the same rows,
// for every format a model admits, at counts that end on short tails; rows() of selected documents and scanners built
// from caller-prepared query weights. Fixed-capacity tiles (#55): slot-by-slot writes equal pack() byte for byte, their
// scans equal block scans bit for bit at every fill (padding and vacated slots ignored), and mapped tile scores equal
// TieredQIVFIndex scores of the same rows, for every metric.
#include "tieredq/tieredq.h"
#include "tieredq/tieredq_blocks.h"
#include <algorithm>
#include <cstring>
#include <numeric>
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
std::vector<float> index_scores(const tq::TieredQ& codec, const float* rows, size_t count, const float* query) {
  tq::TieredQIndex index(codec);
  index.add(rows, count);
  std::vector<float> scores(count);
  for (const auto& hit : index.search_fast(query, count)) scores[hit.id] = hit.score;
  return scores;
}

constexpr size_t tile_fills[] = {1, 17, 100, 255, tq::TieredQFastBlocks::tile_slots};

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Rows land in shuffled slot order, each slot first holding another row it then replaces; a fill's scan skips slots
// past it that hold stale rows.
void check_tiles(const tq::TieredQ& codec, const tq::TieredQFastBlocks& blocks, const std::vector<float>& data,
                 const std::vector<float>& query) {
  constexpr size_t slots = tq::TieredQFastBlocks::tile_slots;
  const size_t stride = codec.bytes_per_vector(), tile_bytes = blocks.tile_bytes();
  require(tile_bytes == blocks.block_bytes(slots) && tile_bytes % tq::TieredQFastBlocks::alignment == 0, "tile size is not an aligned block");
  const auto rows = codec.encode(data.data(), slots);
  std::vector<uint32_t> order(slots); std::iota(order.begin(), order.end(), 0u);
  std::shuffle(order.begin(), order.end(), std::mt19937(11));
  std::vector<uint8_t> buffer(tile_bytes + 1, 0);
  uint8_t* tile = buffer.data() + 1;  // odd address: tiles need no particular alignment
  for (uint32_t slot : order) blocks.write_row(tile, slot, rows.data() + (slots - 1 - slot) * stride);
  for (uint32_t slot : order) blocks.write_row(tile, slot, rows.data() + slot * stride);
  std::vector<uint8_t> packed(tile_bytes);
  blocks.pack(rows.data(), slots, packed.data());
  require(std::equal(packed.begin(), packed.end(), tile), "tile written slot by slot differs from pack()");
  std::vector<uint8_t> unpacked(slots * stride);
  blocks.unpack(tile, slots, unpacked.data());
  require(unpacked == rows, "tile rows differ from the written rows");
  auto scanner = blocks.scanner(query.data());
  for (size_t live : tile_fills) {
    std::vector<uint8_t> block(blocks.block_bytes(live));
    blocks.pack(rows.data(), live, block.data());
    std::vector<float> from_tile(live), from_block(live);
    scanner.score_tile(tile, live, from_tile.data());
    scanner.score(block.data(), live, from_block.data());
    require(same_bits(from_tile, from_block), "tile scores differ from block scores");
  }
  bool rejected = false;
  try { blocks.write_row(tile, slots, rows.data()); } catch (const std::runtime_error&) { rejected = true; }
  require(rejected, "tile slot out of range accepted");
  rejected = false;
  try { std::vector<float> scores(slots + 1); scanner.score_tile(tile, slots + 1, scores.data()); } catch (const std::runtime_error&) { rejected = true; }
  require(rejected, "tile fill out of range accepted");
}

// One coarse list holding every row and a full probe: the IVF scan covers the tile's rows with its own kernels.
void check_tile_ivf_parity(tq::TieredQMetric metric, const std::vector<float>& training, const std::vector<float>& data,
                           const std::vector<float>& query) {
  constexpr size_t live = 200;
  auto config = tq::TieredQConfig::target_code_bits(dimensions, dimensions * 2);
  config.num_centroids = centroids; config.fine_tiers = false; config.metric = metric;
  const auto codec = tq::TieredQ::train(config, training.data(), training_rows);
  tq::TieredQIVFIndex ivf(codec, 1);
  std::vector<float> center(dimensions, 0.0f);
  ivf.set_lists(center.data());
  ivf.build(data.data(), live);
  std::vector<float> expected(live);
  for (const auto& hit : ivf.search(query.data(), live, 1)) expected[hit.id] = hit.score;
  const tq::TieredQFastBlocks blocks(ivf.codec());
  const auto rows = ivf.codec().encode(data.data(), live);
  std::vector<uint8_t> tile(blocks.tile_bytes(), 0);
  for (size_t slot = 0; slot < live; ++slot) blocks.write_row(tile.data(), slot, rows.data() + slot * ivf.codec().bytes_per_vector());
  std::vector<float> scores(live);
  blocks.scanner(query.data()).score_tile(tile.data(), live, scores.data());
  ivf.codec().map_scores(query.data(), scores.data(), live);
  require(same_bits(scores, expected), "mapped tile scores differ from TieredQIVFIndex scores");
}

void check_format(const tq::TieredQ& codec, const tq::TieredQFastBlocks& blocks, const std::vector<float>& data,
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
    // Selected documents, out of order and repeated: each is its packed row.
    std::vector<uint32_t> picked;
    for (size_t doc = count; doc-- > 0;) if (doc % 3 == 0) picked.push_back(uint32_t(doc));
    picked.push_back(0);
    std::vector<uint8_t> selected(picked.size() * stride);
    blocks.rows(block, count, picked.data(), picked.size(), selected.data());
    for (size_t i = 0; i < picked.size(); ++i)
      require(std::equal(rows.begin() + picked[i] * stride, rows.begin() + (picked[i] + 1) * stride, selected.begin() + i * stride),
              "fast block rows differ from the packed rows");
    auto scanner = blocks.scanner(query.data());
    std::vector<float> scores(count), weighted(count);
    scanner.score(block, count, scores.data());
    // Caller-prepared weights scan as the query itself does.
    blocks.scanner(codec.query_weights(query.data())).score(block, count, weighted.data());
    for (size_t doc = 0; doc < count; ++doc)
      require(std::fabs(weighted[doc] - scores[doc]) <= 1e-4f * (1 + std::fabs(scores[doc])), "weight-built scanner differs");
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
    auto config = tq::TieredQConfig::target_code_bits(dimensions, dimensions * bits_per_dim);
    config.num_centroids = centroids; config.fine_tiers = false;
    const auto codec = tq::TieredQ::train(config, training.data(), training_rows);
    const tq::TieredQFastBlocks native(codec);
    check_format(codec, native, data, query, true);
    check_tiles(codec, native, data, query);
    const tq::TieredQFastBlocks reopened(codec, native.format());
    require(reopened.format() == native.format() && reopened.block_bytes(300) == native.block_bytes(300), "persisted format reopens differently");
    for (uint32_t format : candidate_formats) {
      if (format == native.format()) continue;
      try {
        const tq::TieredQFastBlocks portable(codec, format);
        check_format(codec, portable, data, query, false);
        check_tiles(codec, portable, data, query);
      }
      catch (const std::runtime_error& e) { if (!std::string(e.what()).ends_with("does not fit this model") && !std::string(e.what()).ends_with("identify this model's codes")) throw; }
    }
  }
  for (auto metric : {tq::TieredQMetric::inner_product, tq::TieredQMetric::cosine, tq::TieredQMetric::l2})
    check_tile_ivf_parity(metric, training, data, query);
  bool rejected = false;
  try { tq::TieredQFastBlocks(tq::TieredQ::train({dimensions, dimensions, centroids}, training.data(), training_rows), 0); }
  catch (const std::runtime_error&) { rejected = true; }
  require(rejected, "unknown fast block format accepted");
  {
    auto config = tq::TieredQConfig::target_code_bits(dimensions, dimensions * 2);
    config.num_centroids = centroids; config.fine_tiers = false;
    const auto codec = tq::TieredQ::train(config, training.data(), training_rows);
    const tq::TieredQFastBlocks blocks(codec);
    const auto rows = codec.encode(data.data(), 17);
    std::vector<uint8_t> block(blocks.block_bytes(17)), out(codec.bytes_per_vector());
    blocks.pack(rows.data(), 17, block.data());
    const uint32_t outside = 17;
    bool out_of_range = false;
    try { blocks.rows(block.data(), 17, &outside, 1, out.data()); } catch (const std::runtime_error&) { out_of_range = true; }
    require(out_of_range, "block document out of range accepted");
  }
}
