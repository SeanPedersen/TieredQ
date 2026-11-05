// Elias-Fano list encoding, validated decoding and sampled random access (tieredq_elias_fano.h).
#include "tieredq/detail/tieredq_elias_fano.h"
#include "tieredq_io.h"
#include <array>
#include <bit>
#include <climits>
#include <limits>

namespace dv::tieredq_detail {
namespace {
constexpr size_t word_bits = sizeof(uint64_t) * CHAR_BIT;
// Ids between select samples: a lookup scans at most this many ones (and about as many zeros) of the high stream.
constexpr size_t select_interval = 64;

size_t samples_of(uint32_t count) { return (count + select_interval - 1) / select_interval; }
size_t words_of(uint64_t bits) { return size_t((bits + word_bits - 1) / word_bits); }

// width <= 32 bits from bit on; reads the next word only when the field straddles it.
uint32_t read_bits(const std::vector<uint64_t>& words, uint64_t bit, uint32_t width) {
  if (!width) return 0;
  const size_t word = bit / word_bits, offset = bit % word_bits;
  uint64_t field = words[word] >> offset;
  if (offset + width > word_bits) field |= words[word + 1] << (word_bits - offset);
  return uint32_t(field & ((uint64_t(1) << width) - 1));
}

void write_bits(std::vector<uint64_t>& words, uint64_t bit, uint32_t width, uint32_t field) {
  if (!width) return;
  const size_t word = bit / word_bits, offset = bit % word_bits;
  words[word] |= uint64_t(field) << offset;
  if (offset + width > word_bits) words[word + 1] |= uint64_t(field) >> (word_bits - offset);
}

constexpr unsigned byte_values = 1u << CHAR_BIT;
// in_byte[rank * byte_values + byte]: position of the byte's rank-th one.
constexpr auto in_byte = [] {
  std::array<uint8_t, CHAR_BIT * byte_values> table{};
  for (unsigned rank = 0; rank < CHAR_BIT; ++rank)
    for (unsigned byte = 0; byte < byte_values; ++byte)
      for (unsigned bit = 0, seen = 0; bit < CHAR_BIT; ++bit)
        if (byte >> bit & 1 && seen++ == rank) table[rank * byte_values + byte] = uint8_t(bit);
  return table;
}();

// Broadword select (Vigna, "Broadword implementation of rank/select queries"): branch-free, since data-dependent
// loops over the word's ones mispredict and serialize. Requires rank < popcount(ones).
uint64_t select_in_word(uint64_t ones, size_t rank) {
  constexpr uint64_t pairs = 0x5555555555555555, nibbles = 0x3333333333333333, bytes = 0x0F0F0F0F0F0F0F0F;
  constexpr uint64_t ones_step8 = 0x0101010101010101, msbs_step8 = 0x8080808080808080;
  constexpr unsigned top_byte = word_bits - CHAR_BIT, byte_mask = byte_values - 1;
  uint64_t sums = ones - ((ones >> 1) & pairs);
  sums = (sums & nibbles) + ((sums >> 2) & nibbles);
  sums = ((sums + (sums >> 4)) & bytes) * ones_step8;  // byte i: ones in bytes 0..i
  // Bytes whose running count is at most rank: the rank-th one lies past them.
  const uint64_t passed = ((rank * ones_step8 | msbs_step8) - sums) & msbs_step8;
  const unsigned place = unsigned(((passed >> (CHAR_BIT - 1)) * ones_step8) >> top_byte) * CHAR_BIT;
  const unsigned before = unsigned(((sums << CHAR_BIT) >> place) & byte_mask);
  return place + in_byte[(rank - before) * byte_values + ((ones >> place) & byte_mask)];
}
}  // namespace

uint64_t EliasFanoLists::plan(const std::vector<uint32_t>& sizes) {
  universe_ = 0;
  for (uint32_t count : sizes) universe_ += count;
  lists_.assign(sizes.size(), {});
  uint64_t bits = 0;
  size_t samples = 0;
  for (size_t list = 0; list < sizes.size(); ++list) {
    lists_[list] = {bits, sizes[list], uint32_t(samples)};
    const Span placed = span(lists_[list]);
    if (placed.end - placed.high > std::numeric_limits<uint32_t>::max()) bad("Elias-Fano list too long");
    bits = placed.end;
    samples += samples_of(sizes[list]);
  }
  samples_.assign(samples, 0);
  return bits;
}

// The high stream holds count ones plus one zero per bucket up to the universe's largest id.
EliasFanoLists::Span EliasFanoLists::span(const List& entry) const {
  if (!entry.count) return {entry.begin, entry.begin, entry.begin, 0};
  const uint32_t width = uint32_t(std::bit_width(universe_ / entry.count)) - 1;
  const uint64_t high = entry.begin + uint64_t(entry.count) * width;
  return {entry.begin, high, high + entry.count + ((universe_ - 1) >> width) + 1, width};
}

uint64_t EliasFanoLists::select(uint64_t bit, size_t rank, uint64_t end) const {
  if (bit >= end) bad("invalid Elias-Fano ids");
  size_t word = bit / word_bits;
  const size_t last = (end - 1) / word_bits;
  uint64_t ones = words_[word] & (~uint64_t(0) << (bit % word_bits));
  for (size_t found = size_t(std::popcount(ones)); rank >= found; found = size_t(std::popcount(ones))) {
    rank -= found;
    if (++word > last) bad("invalid Elias-Fano ids");
    ones = words_[word];
  }
  const uint64_t position = word * word_bits + select_in_word(ones, rank);
  if (position >= end) bad("invalid Elias-Fano ids");
  return position;
}

uint64_t EliasFanoLists::value(const Span& span, size_t position, uint64_t one) const {
  const uint64_t high = one - span.high - position;
  return (high << span.width) | read_bits(words_, span.low + uint64_t(position) * span.width, span.width);
}

void EliasFanoLists::ids(const uint32_t* lists, const uint32_t* positions, size_t count, uint32_t* out) const {
  struct Pending { Span bits; uint64_t start; size_t rank; };
  std::vector<Pending> pending(count);
  for (size_t i = 0; i < count; ++i) {
    const List& entry = lists_[lists[i]];
    const size_t sample = positions[i] / select_interval;
    pending[i] = {span(entry), 0, positions[i] - sample * select_interval};
    pending[i].start = pending[i].bits.high + samples_[entry.first_sample + sample];
    __builtin_prefetch(words_.data() + pending[i].start / word_bits);
    __builtin_prefetch(words_.data() + (pending[i].bits.low + uint64_t(positions[i]) * pending[i].bits.width) / word_bits);
  }
  for (size_t i = 0; i < count; ++i)
    out[i] = uint32_t(value(pending[i].bits, positions[i], select(pending[i].start, pending[i].rank, pending[i].bits.end)));
}

void EliasFanoLists::index() {
  std::vector<bool> seen(universe_, false);
  for (const List& entry : lists_) {
    const Span bits = span(entry);
    int64_t previous = -1;
    uint64_t one = bits.high;
    for (size_t position = 0; position < entry.count; ++position, ++one) {
      one = select(one, 0, bits.end);
      if (position % select_interval == 0) samples_[entry.first_sample + position / select_interval] = uint32_t(one - bits.high);
      const uint64_t id = value(bits, position, one);
      if (id >= universe_ || int64_t(id) <= previous || seen[id]) bad("invalid Elias-Fano ids");
      seen[id] = true;
      previous = int64_t(id);
    }
  }
}

EliasFanoLists EliasFanoLists::encode(const std::vector<uint32_t>& list_of, size_t lists) {
  std::vector<uint32_t> sizes(lists, 0);
  for (uint32_t list : list_of) {
    if (list >= lists) bad("invalid list id");
    ++sizes[list];
  }
  EliasFanoLists encoded;
  encoded.words_.assign(words_of(encoded.plan(sizes)), 0);
  std::vector<uint32_t> filled(lists, 0);
  for (uint32_t id = 0; id < list_of.size(); ++id) {
    const Span bits = encoded.span(encoded.lists_[list_of[id]]);
    const uint32_t position = filled[list_of[id]]++;
    write_bits(encoded.words_, bits.low + uint64_t(position) * bits.width, bits.width,
               uint32_t(id & ((uint64_t(1) << bits.width) - 1)));
    const uint64_t one = bits.high + (uint64_t(id) >> bits.width) + position;
    encoded.words_[one / word_bits] |= uint64_t(1) << (one % word_bits);
  }
  encoded.index();
  return encoded;
}

EliasFanoLists EliasFanoLists::decode(const std::vector<uint32_t>& sizes, std::vector<uint64_t> words) {
  EliasFanoLists decoded;
  if (words.size() != words_of(decoded.plan(sizes))) bad("invalid Elias-Fano length");
  decoded.words_ = std::move(words);
  decoded.index();
  return decoded;
}

size_t EliasFanoLists::allocated_bytes() const {
  return lists_.capacity() * sizeof(List) + samples_.capacity() * sizeof(uint32_t) + words_.capacity() * sizeof(uint64_t);
}

}  // namespace dv::tieredq_detail
