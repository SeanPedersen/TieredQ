// #33 bridge: trains the public flat index's model and returns its canonical codec rows, so entropy estimates measure
// exactly the symbols TieredQIndex stores, alongside the fast index's real resident bytes.
#include <tieredq/tieredq.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include "training.h"
namespace nb = nanobind;
using Matrix = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

namespace {
constexpr size_t tier_fields = 5;

// Mirrors PythonIndex's constructor (bindings/index.cpp) for fast mode.
tq::TieredQConfig fast_config(uint32_t dimensions, uint32_t code_bits, const std::string& metric, bool fine_tiers,
                              bool exact_fast_scores) {
  auto config = tq::TieredQConfig::target_code_bits(dimensions, code_bits);
  config.target_mode = exact_fast_scores ? tq::TieredQIndexMode::full : tq::TieredQIndexMode::fast;
  config.metric = tieredq_python::parse_metric(metric);
  config.fine_tiers = fine_tiers;
  config.validate();
  return config;
}

template <class T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> owned(std::vector<T> values, size_t rows, size_t columns) {
  auto* data = new std::vector<T>(std::move(values));
  nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  return nb::ndarray<nb::numpy, T, nb::ndim<2>>(data->data(), {rows, columns}, owner);
}

nb::dict train_encode(Matrix data, uint32_t code_bits, const std::string& metric, bool fine_tiers,
                      bool exact_fast_scores) {
  const size_t count = data.shape(0), dimensions = data.shape(1);
  std::vector<uint8_t> rows;
  std::vector<uint32_t> tiers;
  size_t row_bytes = 0, fast_storage = 0, fast_stride = 0, model_bytes = 0;
  uint32_t realised_bits = 0, prefix_bits = 0;
  {
    nb::gil_scoped_release release;
    auto codec = tieredq_python::train_codec(
        fast_config(uint32_t(dimensions), code_bits, metric, fine_tiers, exact_fast_scores), data.data(), count);
    rows = codec.encode(data.data(), count);
    row_bytes = codec.bytes_per_vector();
    realised_bits = codec.config().code_bits;
    prefix_bits = codec.fast_prefix_bits();
    model_bytes = codec.serialize().size();
    for (const auto& tier : codec.tiers())
      tiers.insert(tiers.end(), {tier.start, tier.dimensions, tier.bits, tier.scale_group, tier.bit_offset});
    tq::TieredQIndex index(std::move(codec));
    index.add(data.data(), count);
    fast_storage = index.storage_bytes();
    fast_stride = index.bytes_per_vector();
  }
  nb::dict out;
  const size_t tier_count = tiers.size() / tier_fields;
  out["tiers"] = owned(std::move(tiers), tier_count, tier_fields);
  out["rows"] = owned(std::move(rows), count, row_bytes);
  out["code_bits"] = realised_bits;
  out["prefix_bits"] = prefix_bits;
  out["model_bytes"] = model_bytes;
  out["fast_storage_bytes"] = fast_storage;
  out["fast_bytes_per_vector"] = fast_stride;
  return out;
}
}  // namespace

NB_MODULE(compact_native, m) {
  m.def("train_encode", &train_encode, nb::arg("data"), nb::arg("code_bits"), nb::arg("metric") = "ip",
        nb::arg("fine_tiers") = true, nb::arg("exact_fast_scores") = false);
}
