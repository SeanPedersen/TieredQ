// TieredQFastBlocks: fast storage serialized into caller blocks and scanned in place through borrowed views.
// Block layout for count documents, padded to the format's block granularity (16 documents; nibble lookups keep
// 256-document blocks and end on a tail block rounded to 16 documents):
//   factor u16[padded] | correction u16[padded] | centroid u8[padded] | 64-byte aligned codes | read slack
#include "tieredq/tieredq_blocks.h"
#include "tieredq_fast_scan.h"
#include "tieredq_fast_rows.h"
#include <cstring>
#include <numeric>

namespace dv {
using tieredq_detail::bad;
using tieredq_detail::FastScanCache;
namespace {
constexpr uint32_t lookup_format_flag = 2, levels_format_flag = 1, format_level_shift = 8, format_flags = 0xff;
// Vector loads of the last pair row may run up to one 64-document chunk past a short tail.
constexpr size_t read_slack = tieredq_detail::lookup_chunk_docs;

size_t round_up(size_t value, size_t multiple) { return (value + multiple - 1) / multiple * multiple; }

// Nibble lookups: the flag alone (their tables carry no level range); levels and code words: range << 8 | flag.
uint32_t format_of(const FastScanCache& cache) {
  return cache.lookup ? lookup_format_flag : uint32_t(cache.level_max) << format_level_shift | levels_format_flag;
}

struct Shape {
  size_t blocks = 0, padded = 0, tail_block = SIZE_MAX, tail_docs = 0, code_offset = 0, code_bytes = 0, total = 0;
};

Shape shape_of(const FastScanCache& geometry, size_t count) {
  using namespace tieredq_detail;
  Shape s;
  if (geometry.lookup) {
    const size_t full = count / lookup_block_docs, rest = count % lookup_block_docs;
    s.blocks = full + (rest ? 1 : 0);
    s.tail_docs = round_up(rest, lookup_vector_docs);
    if (rest) s.tail_block = full;
    s.padded = full * lookup_block_docs + s.tail_docs;
    s.code_bytes = s.padded * geometry.lookup->pairs();
  } else {
    s.blocks = (count + fast_scan_block_docs - 1) / fast_scan_block_docs;
    s.padded = s.blocks * fast_scan_block_docs;
    s.code_bytes = s.blocks * geometry.level_block_bytes();
  }
  s.code_offset = round_up(s.padded * (2 * sizeof(uint16_t) + sizeof(uint8_t)), TieredQFastBlocks::alignment);
  s.total = s.code_offset + s.code_bytes + read_slack;
  return s;
}

// Points view (a copy of the format's geometry) at count documents of block.
void borrow(FastScanCache& view, const uint8_t* block, size_t count) {
  const Shape s = shape_of(view, count);
  view.count = count; view.blocks = s.blocks;
  const auto* halves = reinterpret_cast<const uint16_t*>(block);
  view.factor.borrow(halves, s.padded);
  view.correction.borrow(halves + s.padded, s.padded);
  view.centroid.borrow(block + s.padded * 2 * sizeof(uint16_t), s.padded);
  if (view.lookup) {
    view.lookup->codes.borrow(block + s.code_offset, s.code_bytes);
    view.lookup->tail_block = s.tail_block; view.lookup->tail_docs = s.tail_docs;
  } else {
    view.levels.borrow(reinterpret_cast<const int8_t*>(block + s.code_offset), s.code_bytes);
  }
}
}  // namespace

struct TieredQFastBlocks::Scanner::Impl {
  virtual ~Impl() = default;
  virtual void score(const uint8_t* block, size_t count, float* scores) = 0;
};

namespace {
template<class Prepared, class Scan>
struct TypedScanner final : TieredQFastBlocks::Scanner::Impl {
  TypedScanner(std::unique_ptr<FastScanCache> view, Prepared prepared, Scan scan)
      : view_(std::move(view)), prepared_(std::move(prepared)), scan_(std::move(scan)) {}
  void score(const uint8_t* block, size_t count, float* scores) override {
    if (!count) return;
    if (!block || !scores) bad("null block buffer");
    borrow(*view_, block, count);
    // Kernels score whole vectors of documents; padding lanes land in scratch.
    scratch_.resize(view_->blocks * (view_->lookup ? tieredq_detail::lookup_block_docs : tieredq_detail::fast_scan_block_docs));
    scan_(prepared_.data(), 1, 0, count, scratch_.data(), scratch_.size());
    std::copy_n(scratch_.data(), count, scores);
  }
  std::unique_ptr<FastScanCache> view_;  // heap address stays fixed: scan_ refers to it
  Prepared prepared_;
  Scan scan_;
  std::vector<float> scratch_;
};
}  // namespace

TieredQFastBlocks::Scanner::Scanner(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TieredQFastBlocks::Scanner::~Scanner() = default;
TieredQFastBlocks::Scanner::Scanner(Scanner&&) noexcept = default;
TieredQFastBlocks::Scanner& TieredQFastBlocks::Scanner::operator=(Scanner&&) noexcept = default;
void TieredQFastBlocks::Scanner::score(const uint8_t* block, size_t count, float* scores) { impl_->score(block, count, scores); }

TieredQFastBlocks::TieredQFastBlocks(const TieredQ& codec) : codec_(codec) {
  const TieredQIndex index(codec_, TieredQIndexMode::full);
  empty_ = index.make_fast_storage(nullptr, 0);
  format_ = format_of(*empty_);
}

TieredQFastBlocks::TieredQFastBlocks(const TieredQ& codec, uint32_t format) : codec_(codec), format_(format) {
  using namespace tieredq_detail;
  const bool lookup = (format & format_flags) == lookup_format_flag;
  const int level_max = int(format >> format_level_shift);
  if (!lookup && (format & format_flags) != levels_format_flag) bad("unknown fast block format");
  if (lookup ? !(fast_lookup_supported(codec_.tiers_, codec_.levels_) && level_max == 0) :
      level_max == fast_code_lookup_levels ? !fast_code_lookup_supported(codec_.tiers_) :
      level_max != x86_fast_level_max && level_max != arm_fast_level_max)
    bad("fast block format does not fit this model");
  const TieredQIndex index(codec_, TieredQIndexMode::full);
  empty_ = index.make_fast_storage(nullptr, 0, lookup, lookup ? x86_fast_level_max : level_max);
  if (!empty_->lookup && !empty_->code_lookup())
    for (const auto& codes : fast_rows::level_inverse(*empty_, codec_.levels_, codec_.tiers_))
      for (int16_t code : codes) if (code == fast_rows::ambiguous_code) bad("fast block levels cannot identify this model's codes");
}

size_t TieredQFastBlocks::block_bytes(size_t count) const { return shape_of(*empty_, count).total; }

void TieredQFastBlocks::pack(const uint8_t* rows, size_t count, uint8_t* block) const {
  using namespace tieredq_detail;
  if (!block) bad("null block buffer");
  const Shape s = shape_of(*empty_, count);
  std::memset(block, 0, s.total);
  if (!count) return;
  if (!rows) bad("null row buffer");
  const TieredQIndex index(codec_, TieredQIndexMode::full);
  const auto cache = index.make_fast_storage(rows, count, empty_->lookup != nullptr,
                                             empty_->lookup ? tieredq_detail::x86_fast_level_max : empty_->level_max);
  std::memcpy(block, cache->factor.data(), s.padded * sizeof(uint16_t));
  std::memcpy(block + s.padded * sizeof(uint16_t), cache->correction.data(), s.padded * sizeof(uint16_t));
  std::memcpy(block + s.padded * 2 * sizeof(uint16_t), cache->centroid.data(), s.padded);
  uint8_t* codes = block + s.code_offset;
  if (!cache->lookup) { std::memcpy(codes, cache->levels.data(), s.code_bytes); return; }
  // Whole 256-document blocks copy as they are; the tail keeps tail_docs of each pair row.
  const size_t pairs = cache->lookup->pairs(), whole = s.tail_block == SIZE_MAX ? s.blocks : s.tail_block;
  const uint8_t* source = cache->lookup->codes.data();
  std::memcpy(codes, source, whole * pairs * lookup_block_docs);
  if (s.tail_block == SIZE_MAX) return;
  const uint8_t* tail = source + whole * pairs * lookup_block_docs;
  uint8_t* out = codes + whole * pairs * lookup_block_docs;
  for (size_t pair = 0; pair < pairs; ++pair) std::memcpy(out + pair * s.tail_docs, tail + pair * lookup_block_docs, s.tail_docs);
}

void TieredQFastBlocks::unpack(const uint8_t* block, size_t count, uint8_t* rows) const {
  if (!count) return;
  if (!block || !rows) bad("null block buffer");
  const auto view = tieredq_detail::copy_fast_storage(*empty_);
  borrow(*view, block, count);
  std::vector<uint32_t> docs(count); std::iota(docs.begin(), docs.end(), 0u);
  tieredq_detail::read_fast_rows(*view, codec_.levels_, codec_.tiers_, codec_.coded_dimensions(), codec_.code_bytes(),
                                 codec_.bytes_per_vector(), docs.data(), count, rows);
}

TieredQFastBlocks::Scanner TieredQFastBlocks::scanner(const float* query) const {
  if (!query) bad("null query");
  std::vector<float> lifted;
  const float* model_query = query;
  if (codec_.config().metric != TieredQMetric::inner_product) {
    lifted.resize(codec_.model_dimensions());
    codec_.lift_queries(query, 1, lifted.data());
    model_query = lifted.data();
  }
  auto view = std::unique_ptr<FastScanCache>(new FastScanCache(*tieredq_detail::copy_fast_storage(*empty_)));
  std::unique_ptr<Scanner::Impl> impl;
  FastScanCache& geometry = *view;
  tieredq_detail::with_fast_scan(geometry, codec_, model_query, 1, [&](auto&& prepared, size_t, auto scan) {
    using Typed = TypedScanner<std::decay_t<decltype(prepared)>, decltype(scan)>;
    impl = std::make_unique<Typed>(std::move(view), std::move(prepared), std::move(scan));
  });
  return Scanner(std::move(impl));
}
}  // namespace dv
