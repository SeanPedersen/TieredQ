// Array ownership and shape validation shared by the Python adapters.
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <tieredq/tieredq.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tieredq_python {
namespace nb = nanobind;
using FloatArray = nb::ndarray<nb::numpy, const float, nb::c_contig, nb::device::cpu>;
using Matrix = nb::ndarray<nb::numpy, const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template<class T>
nb::ndarray<nb::numpy, T> owned_array(std::vector<T> data, const std::vector<size_t>& shape) {
  auto* storage = new std::vector<T>(std::move(data));
  nb::capsule owner(storage, [](void* pointer) noexcept { delete static_cast<std::vector<T>*>(pointer); });
  return nb::ndarray<nb::numpy, T>(storage->data(), shape.size(), shape.data(), owner);
}
template<class T>
nb::ndarray<nb::numpy, T> owned_array(std::vector<T> data, std::initializer_list<size_t> shape) {
  return owned_array(std::move(data), std::vector<size_t>(shape));
}
inline void require_width(size_t actual, size_t expected) {
  if (actual != expected) throw std::invalid_argument("array width does not match index");
}
inline size_t array_elements(size_t rows, size_t columns) {
  if (columns && rows > std::numeric_limits<size_t>::max() / columns)
    throw std::invalid_argument("array size overflow");
  return rows * columns;
}
void bind_index(nb::module_& module);
}  // namespace tieredq_python
