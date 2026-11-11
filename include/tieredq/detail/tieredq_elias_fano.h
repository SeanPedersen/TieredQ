// Ascending ids of disjoint lists that partition [0, universe) as Elias-Fano sequences (#34): a list of n ids keeps n
// low parts of L = floor(log2(universe / n)) bits each, then the high parts as n ones in a unary bit stream. All lists
// share one stream whose layout follows from the list sizes alone, so persistence stores only the words. A select
// sample every select_interval ids bounds each random access to a short scan.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tq::tieredq_detail {

class EliasFanoLists {
 public:
  EliasFanoLists() = default;
  // list_of[id] is the list of every id in [0, list_of.size()); each list holds its ids in ascending order.
  static EliasFanoLists encode(const std::vector<uint32_t>& list_of, size_t lists);
  // The lists of sizes back from encode()'s words(); throws unless they encode a partition of [0, sum of sizes).
  static EliasFanoLists decode(const std::vector<uint32_t>& sizes, std::vector<uint64_t> words);
  // The ids at count (list, position) pairs, each position inside its list, in stages so that the memory accesses of
  // different pairs overlap.
  void ids(const uint32_t* lists, const uint32_t* positions, size_t count, uint32_t* out) const;
  // visit(position, id) over one list in ascending order.
  template<class Visit> void for_each(size_t list, Visit visit) const {
    const Span bits = span(lists_[list]);
    uint64_t one = bits.high;
    for (size_t position = 0; position < lists_[list].count; ++position, ++one) {
      one = select(one, 0, bits.end);
      visit(position, uint32_t(value(bits, position, one)));
    }
  }
  const std::vector<uint64_t>& words() const { return words_; }
  size_t allocated_bytes() const;

 private:
  struct List {
    uint64_t begin = 0;         // first bit: count low parts, then the high stream
    uint32_t count = 0;
    uint32_t first_sample = 0;  // index into samples_
  };
  // A list's bit ranges: low parts from low, unary high parts in [high, end).
  struct Span {
    uint64_t low, high, end;
    uint32_t width;  // bits per low part
  };
  // Sets the universe and every list's place in the stream; returns the stream's length in bits.
  uint64_t plan(const std::vector<uint32_t>& sizes);
  Span span(const List& entry) const;
  // Absolute position of the rank-th one (0-based) at or after bit, below end; throws when there is none.
  uint64_t select(uint64_t bit, size_t rank, uint64_t end) const;
  // The id at position whose high-stream one sits at bit one; wider than an id, so corrupt streams cannot wrap.
  uint64_t value(const Span& span, size_t position, uint64_t one) const;
  // Fills samples_ from the stream and validates it as a partition of the universe.
  void index();
  std::vector<List> lists_;
  std::vector<uint32_t> samples_;  // high-stream offset of every select_interval-th one per list
  std::vector<uint64_t> words_;
  size_t universe_ = 0;
};

}  // namespace tq::tieredq_detail
