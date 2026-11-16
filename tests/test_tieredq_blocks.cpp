// TieredQFastBlocks: exact pack/unpack round trips and in-place scores equal to the fast index over the same rows,
// for every format a model admits, at counts that end on short tails; rows() of selected documents and scanners built
// from caller-prepared query weights. Partly filled blocks (#55): slot-by-slot writes equal pack() byte for byte at
// every capacity, scans of the first live slots equal block scans of those rows bit for bit (stale slots ignored), and
// mapped block scores equal TieredQIVFIndex scores of the same rows, for every metric; reset scanners score as fresh ones.
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

// Capacities that end on short nibble-lookup tails (17, 100, 300, 513) and on whole blocks (256).
constexpr size_t capacities[] = {1, 17, 100, 256, 300, 513};

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Rows land in shuffled slot order, each slot first holding another row it then replaces; scans of the first `live`
// slots skip slots past them that hold stale rows.
void check_partial_blocks(const tq::TieredQ& codec, const tq::TieredQFastBlocks& blocks, const std::vector<float>& data,
                          const std::vector<float>& query) {
  const size_t stride = codec.bytes_per_vector();
  require(blocks.block_bytes(tq::TieredQFastBlocks::tile_slots) % tq::TieredQFastBlocks::alignment == 0, "tile size is not aligned");
  auto scanner = blocks.scanner(query.data());
  for (size_t capacity : capacities) {
    const auto rows = codec.encode(data.data(), capacity);
    std::vector<uint32_t> order(capacity); std::iota(order.begin(), order.end(), 0u);
    std::shuffle(order.begin(), order.end(), std::mt19937(11));
    std::vector<uint8_t> buffer(blocks.block_bytes(capacity) + 1, 0);
    uint8_t* block = buffer.data() + 1;  // odd address: blocks need no particular alignment
    for (uint32_t slot : order) blocks.write_row(block, capacity, slot, rows.data() + (capacity - 1 - slot) * stride);
    for (uint32_t slot : order) blocks.write_row(block, capacity, slot, rows.data() + slot * stride);
    std::vector<uint8_t> packed(blocks.block_bytes(capacity));
    blocks.pack(rows.data(), capacity, packed.data());
    require(std::equal(packed.begin(), packed.end(), block), "block written slot by slot differs from pack()");
    std::vector<uint8_t> unpacked(capacity * stride);
    blocks.unpack(block, capacity, unpacked.data());
    require(unpacked == rows, "written rows differ from the rows read back");
    for (size_t live : {size_t(1), capacity / 3 + 1, capacity - 1, capacity}) {
      if (!live) continue;
      std::vector<uint8_t> exact(blocks.block_bytes(live));
      blocks.pack(rows.data(), live, exact.data());
      std::vector<float> partial(live), whole(live);
      scanner.score(block, capacity, live, partial.data());
      scanner.score(exact.data(), live, whole.data());
      require(same_bits(partial, whole), "partly filled block scores differ from a block of its live rows");
    }
    bool rejected = false;
    try { blocks.write_row(block, capacity, capacity, rows.data()); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "block slot out of range accepted");
    rejected = false;
    try { std::vector<float> scores(capacity + 1); scanner.score(block, capacity, capacity + 1, scores.data()); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "block live count past capacity accepted");
  }
}

// A scanner reset to another query (from the query or its weights) scores as a fresh scanner of that query, bit for bit.
void check_reset(const tq::TieredQ& codec, const tq::TieredQFastBlocks& blocks, const std::vector<float>& data,
                 const std::vector<float>& query) {
  constexpr size_t count = 300;
  const auto rows = codec.encode(data.data(), count);
  std::vector<uint8_t> block(blocks.block_bytes(count));
  blocks.pack(rows.data(), count, block.data());
  const float* other = data.data() + (count + 1) * dimensions;
  std::vector<float> fresh(count), reset(count), weighted(count);
  blocks.scanner(other).score(block.data(), count, fresh.data());
  auto scanner = blocks.scanner(query.data());
  scanner.score(block.data(), count, reset.data());
  blocks.reset(scanner, other);
  scanner.score(block.data(), count, reset.data());
  require(same_bits(fresh, reset), "reset scanner differs from a fresh scanner");
  auto from_weights = blocks.scanner(codec.query_weights(query.data()));
  blocks.reset(from_weights, codec.query_weights(other));
  from_weights.score(block.data(), count, weighted.data());
  std::vector<float> fresh_weighted(count);
  blocks.scanner(codec.query_weights(other)).score(block.data(), count, fresh_weighted.data());
  require(same_bits(fresh_weighted, weighted), "scanner reset from weights differs from a fresh one");
  bool rejected = false;
  const tq::TieredQFastBlocks other_blocks(codec);
  try { other_blocks.reset(scanner, other); } catch (const std::runtime_error&) { rejected = true; }
  require(rejected, "scanner reset by other fast blocks accepted");
}

// One coarse list holding every row and a full probe: the IVF scan covers the block's rows with its own kernels.
void check_block_ivf_parity(tq::TieredQMetric metric, const std::vector<float>& training, const std::vector<float>& data,
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
  constexpr size_t capacity = tq::TieredQFastBlocks::tile_slots;
  std::vector<uint8_t> tile(blocks.block_bytes(capacity), 0);
  for (size_t slot = 0; slot < live; ++slot) blocks.write_row(tile.data(), capacity, slot, rows.data() + slot * ivf.codec().bytes_per_vector());
  std::vector<float> scores(live);
  blocks.scanner(query.data()).score(tile.data(), capacity, live, scores.data());
  ivf.codec().map_scores(query.data(), scores.data(), live);
  require(same_bits(scores, expected), "mapped block scores differ from TieredQIVFIndex scores");
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
    check_partial_blocks(codec, native, data, query);
    check_reset(codec, native, data, query);
    const tq::TieredQFastBlocks reopened(codec, native.format());
    require(reopened.format() == native.format() && reopened.block_bytes(300) == native.block_bytes(300), "persisted format reopens differently");
    for (uint32_t format : candidate_formats) {
      if (format == native.format()) continue;
      try {
        const tq::TieredQFastBlocks portable(codec, format);
        check_format(codec, portable, data, query, false);
        check_partial_blocks(codec, portable, data, query);
        check_reset(codec, portable, data, query);
      }
      catch (const std::runtime_error& e) { if (!std::string(e.what()).ends_with("does not fit this model") && !std::string(e.what()).ends_with("identify this model's codes")) throw; }
    }
  }
  for (auto metric : {tq::TieredQMetric::inner_product, tq::TieredQMetric::cosine, tq::TieredQMetric::l2})
    check_block_ivf_parity(metric, training, data, query);
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
