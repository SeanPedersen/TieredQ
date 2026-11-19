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
// Search outputs, shaped rows x width, or width for a single 1-D query.
struct SearchResult {
  std::vector<float> scores;
  std::vector<uint32_t> ids;
  std::vector<size_t> shape;
};
struct QueryRows {
  size_t rows;
  bool single;  // a 1-D query, answered with 1-D outputs
};
inline QueryRows query_rows(const FloatArray& queries, size_t dimensions) {
  const size_t rank = queries.ndim();
  if (rank != 1 && rank != 2) throw std::invalid_argument("queries must be a 1-D or 2-D float32 array");
  require_width(queries.shape(rank - 1), dimensions);
  return {rank == 1 ? 1 : queries.shape(0), rank == 1};
}
// Ranking-trained query projection. The codec checks shapes; stored rows are not touched.
inline tq::TieredQ::QueryDecoder make_query_decoder(const Matrix& basis, const Matrix& centroids,
                                                    const Matrix& corr_dirs, const FloatArray& scale,
                                                    const FloatArray& offset) {
  if (scale.ndim() != 1 || offset.ndim() != 1)
    throw std::invalid_argument("scale and offset must be 1-D float32 arrays");
  tq::TieredQ::QueryDecoder decoder;
  decoder.basis.assign(basis.data(), basis.data() + basis.size());
  decoder.centroids.assign(centroids.data(), centroids.data() + centroids.size());
  decoder.corr_dirs.assign(corr_dirs.data(), corr_dirs.data() + corr_dirs.size());
  decoder.scale.assign(scale.data(), scale.data() + scale.size());
  decoder.offset.assign(offset.data(), offset.data() + offset.size());
  return decoder;
}
inline constexpr char query_decoder_doc[] =
    "Optional ranking-trained query projection (basis d x m, centroids and corr_dirs centroids x d,\n"
    "scale and offset length m). Stored codes, lists and tier rows stay unchanged; clear_query_decoder()\n"
    "restores the original projection. Not saved with the index. Helps at low bit budgets only.";

inline SearchResult empty_result(QueryRows queries, size_t width) {
  const size_t elements = array_elements(queries.rows, width);
  return {std::vector<float>(elements, 0.0f), std::vector<uint32_t>(elements, 0),
          queries.single ? std::vector<size_t>{width} : std::vector<size_t>{queries.rows, width}};
}
// Needs the GIL.
inline nb::tuple result_tuple(SearchResult result) {
  return nb::make_tuple(owned_array(std::move(result.scores), result.shape),
                        owned_array(std::move(result.ids), result.shape));
}
void bind_index(nb::module_& module);
void bind_ivf(nb::module_& module);
void bind_ivf_flat(nb::module_& module);
void bind_nested(nb::module_& module);
}  // namespace tieredq_python
