// Immutable model arrays shared between copies of a model: a copied TieredQ (e.g. a fine-tier index's prefix view of
// its model) reads the same basis and centroids instead of holding its own. Arrays are set whole, never modified.
#pragma once
#include <cstddef>
#include <memory>
#include <vector>

namespace dv::tieredq_detail {

template<class T> class SharedArray {
 public:
  SharedArray() = default;
  SharedArray& operator=(std::vector<T> values) {
    values_ = std::make_shared<const std::vector<T>>(std::move(values));
    return *this;
  }
  const std::vector<T>& get() const { return values_ ? *values_ : none(); }
  operator const std::vector<T>&() const { return get(); }
  size_t size() const { return get().size(); }
  bool empty() const { return get().empty(); }
  const T* data() const { return get().data(); }
  const T& operator[](size_t i) const { return get()[i]; }
  auto begin() const { return get().begin(); }
  auto end() const { return get().end(); }
  size_t bytes() const { return get().capacity() * sizeof(T); }
  // True when both hold the same allocation (not merely equal values).
  bool shares(const SharedArray& other) const { return values_ && values_ == other.values_; }

 private:
  static const std::vector<T>& none() {
    static const std::vector<T> none;
    return none;
  }
  std::shared_ptr<const std::vector<T>> values_;
};

}  // namespace dv::tieredq_detail
