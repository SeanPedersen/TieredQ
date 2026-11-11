// Fast-storage array that owns its elements or borrows a caller's read-only buffer (TieredQFastBlocks scans blocks in
// caller memory without copying). Readers see one interface; mutation is valid only while the array owns its data.
#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tq::tieredq_detail {
template<class T>
class BlockArray {
 public:
  using value_type = T;
  BlockArray() = default;
  BlockArray(std::vector<T> owned) : owned_(std::move(owned)) {}
  const T* data() const { return borrowed_ ? borrowed_ : owned_.data(); }
  T* data() { return mutable_owned().data(); }
  size_t size() const { return borrowed_ ? borrowed_size_ : owned_.size(); }
  bool empty() const { return size()==0; }
  size_t max_size() const { return owned_.max_size(); }
  const T& operator[](size_t i) const { return data()[i]; }
  T& operator[](size_t i) { return mutable_owned()[i]; }
  const T* begin() const { return data(); }
  const T* end() const { return data()+size(); }
  // Capacity of owned storage only: borrowed bytes belong to the caller.
  size_t capacity() const { return borrowed_ ? 0 : owned_.capacity(); }
  void reserve(size_t n) { mutable_owned().reserve(n); }
  void resize(size_t n,T value=T()) { mutable_owned().resize(n,value); }
  void assign(size_t n,T value) { mutable_owned().assign(n,value); }
  void clear() { borrowed_=nullptr; borrowed_size_=0; owned_.clear(); }
  void push_back(T value) { mutable_owned().push_back(value); }
  // Points readers at count caller-owned elements, which must outlive every read; drops owned storage.
  void borrow(const T* data,size_t count) { owned_.clear(); owned_.shrink_to_fit(); borrowed_=data; borrowed_size_=count; }
  const std::vector<T>& owned() const { return owned_; }

 private:
  std::vector<T>& mutable_owned() {
    if (borrowed_) throw std::logic_error("TieredQ: borrowed fast storage is read-only");
    return owned_;
  }
  std::vector<T> owned_;
  const T* borrowed_ = nullptr;
  size_t borrowed_size_ = 0;
};
} // namespace tq::tieredq_detail
