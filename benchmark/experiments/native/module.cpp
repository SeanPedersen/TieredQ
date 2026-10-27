// Shared experimental bridge: train immutable codecs and evaluate native index representations.
#include <tieredq/tieredq.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <omp.h>
namespace nb = nanobind;
using Matrix = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Output = nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Ids = nb::ndarray<uint32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
#ifndef EXPERIMENT_MODULE
#define EXPERIMENT_MODULE codebook_native
#endif
#ifdef EXPERIMENT_PROFILE
#include "profile.h"
#endif
namespace {
constexpr unsigned experiment_centroids = 256;
std::vector<uint8_t> train_total(Matrix sample, unsigned total_bits, bool rotate, uint64_t seed) {
  auto config = dv::TieredQConfig::target_code_bits(sample.shape(1), total_bits);
  config.rotate_tiers = rotate;
  config.num_centroids = experiment_centroids;
  nb::gil_scoped_release release;
  return dv::TieredQ::train(config, sample.data(), sample.shape(0), seed).serialize();
}
}
NB_MODULE(EXPERIMENT_MODULE, m) {
  m.def("threads", [](int n) { omp_set_num_threads(n); });
  m.def("train_total", &train_total);
  m.def("train", [](Matrix sample, unsigned bits, unsigned original_d, bool rotate, uint64_t seed) {
    return train_total(sample, bits * original_d, rotate, seed);
  });
#ifdef EXPERIMENT_PROFILE
  m.def("profile_reset", &experiment_profile::reset);
  m.def("profile_read", &experiment_profile::read);
#endif
  nb::enum_<dv::TieredQIndexMode>(m, "Mode")
    .value("fast", dv::TieredQIndexMode::fast)
    .value("full", dv::TieredQIndexMode::full);
  nb::class_<dv::TieredQIndex>(m, "Index")
    .def(nb::init<dv::TieredQ, dv::TieredQIndexMode>())
    .def_static("load", &dv::TieredQIndex::load)
    .def("save", &dv::TieredQIndex::save)
    .def("stride", &dv::TieredQIndex::bytes_per_vector)
    .def("scalar_fast", &dv::TieredQIndex::uses_scalar_fast_scan)
    .def("add", [](dv::TieredQIndex& index, Matrix base) {
      nb::gil_scoped_release release; index.add(base.data(), base.shape(0));
    })
    .def("search", [](const dv::TieredQIndex& index, Matrix q, Output scores, Ids ids) {
      if (q.shape(1)!=index.codec().config().dimensions || ids.shape(1)>index.size())
        throw std::invalid_argument("query width or k mismatch");
      if (q.shape(0)!=scores.shape(0) || q.shape(0)!=ids.shape(0) || scores.shape(1)!=ids.shape(1))
        throw std::invalid_argument("output shape mismatch");
      nb::gil_scoped_release release;
      if (index.mode()==dv::TieredQIndexMode::fast) index.search_batch_fast(q.data(), q.shape(0), ids.shape(1), scores.data(), ids.data());
      else index.search_batch(q.data(), q.shape(0), ids.shape(1), scores.data(), ids.data());
    })
    .def("storage", &dv::TieredQIndex::storage_bytes)
    .def("model_memory", [](const dv::TieredQIndex& i) { return i.codec().memory_bytes(); });
  nb::class_<dv::TieredQ>(m, "Codec")
    .def_static("load", &dv::TieredQ::deserialize)
    .def("roundtrip", &dv::TieredQ::serialize)
    .def("reconstruct", [](const dv::TieredQ& codec, Matrix input, Output out) {
      if (input.shape(0)!=out.shape(0) || input.shape(1)!=out.shape(1)) throw std::invalid_argument("reconstruction shape mismatch");
      nb::gil_scoped_release release;
      std::vector<uint8_t> payload(input.shape(0)*codec.bytes_per_vector());
      codec.encode(input.data(), input.shape(0), payload.data());
      codec.decode(payload.data(), input.shape(0), out.data());
    });
}
