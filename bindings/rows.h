// Build inputs shared by the IVF and nested adapters, as one row loader: a float32 matrix, an SQ8 store (int8 codes,
// e.g. a numpy memmap, with per-dimension float32 scales and offsets; row = codes * scales + offsets, decoded in the
// library a chunk at a time, so a large base needs no float copy), or a Python callable returning rows by id.
// Loaders run on the building thread with the GIL released; the callable reacquires it per chunk.
#pragma once
#include "arrays.h"
#include <nanobind/stl/optional.h>
#include <tieredq/tieredq_nested.h>
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <vector>

namespace tieredq_python {
using CodeMatrix = nb::ndarray<nb::numpy, const int8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using FloatVector = nb::ndarray<nb::numpy, const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using OptionalMatrix = std::optional<Matrix>;
using OptionalCodes = std::optional<CodeMatrix>;
using OptionalVector = std::optional<FloatVector>;
using OptionalLoader = std::optional<nb::callable>;
using OptionalCount = std::optional<size_t>;

struct RowInput {
  tq::TieredQIVFIndex::RowLoader loader;
  size_t count = 0;
};

inline std::vector<float> float_values(const FloatVector& values) {
  return std::vector<float>(values.data(), values.data() + values.shape(0));
}
inline RowInput float_rows(const Matrix& rows, size_t width) {
  require_width(rows.shape(1), width);
  const float* data = rows.data();
  return {[data, width](const uint32_t* ids, size_t count, float* out) {
            for (size_t i = 0; i < count; ++i) std::copy_n(data + size_t(ids[i]) * width, width, out + i * width);
          }, rows.shape(0)};
}
inline RowInput sq8_rows(const CodeMatrix& codes, const FloatVector& scales, const FloatVector& offsets, size_t width) {
  require_width(codes.shape(1), width);
  if (scales.shape(0) != width || offsets.shape(0) != width)
    throw std::invalid_argument("scales and offsets need one value per dimension");
  return {tq::sq8_row_loader(codes.data(), codes.shape(0), float_values(scales), float_values(offsets)), codes.shape(0)};
}
// loader(ids: uint32 array) returns len(ids) x width float32 rows. The handle is borrowed: the callable is an argument
// of the binding call that runs the build, so it outlives every copy of the loader.
inline RowInput callable_rows(nb::handle loader, size_t count, size_t width) {
  return {[loader, width](const uint32_t* ids, size_t n, float* out) {
            nb::gil_scoped_acquire gil;
            nb::object result = loader(owned_array(std::vector<uint32_t>(ids, ids + n), {n}));
            const auto rows = nb::cast<Matrix>(result);
            if (rows.shape(0) != n || rows.shape(1) != width)
              throw std::invalid_argument("loader must return len(ids) x dimensions float32 rows");
            std::copy_n(rows.data(), n * width, out);
          }, count};
}
// Exactly one of rows, (codes, scales, offsets) or (loader, count).
inline RowInput row_input(const OptionalMatrix& rows, const OptionalCodes& codes, const OptionalVector& scales,
                          const OptionalVector& offsets, const OptionalLoader& loader, OptionalCount count,
                          size_t width) {
  const int sources = int(rows.has_value()) + int(codes.has_value()) + int(loader.has_value());
  if (sources != 1) throw std::invalid_argument("pass exactly one of rows, codes= (with scales=, offsets=) or loader=");
  if (rows) return float_rows(*rows, width);
  if (codes) {
    if (!scales || !offsets) throw std::invalid_argument("codes= needs scales= and offsets=");
    return sq8_rows(*codes, *scales, *offsets, width);
  }
  if (!count) throw std::invalid_argument("loader= needs count=");
  return callable_rows(*loader, *count, width);
}
}  // namespace tieredq_python
