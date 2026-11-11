// Python classes for the IVF's coarse list training and two-level router settings.
#include "lists.h"
#include <nanobind/stl/string.h>
#include <string>

namespace tieredq_python {
using namespace nb::literals;

void bind_lists(nb::module_& module) {
  using Training = dv::TieredQCoarseTraining;
  const Training exact{};
  nb::class_<Training>(module, "TieredQCoarseTraining",
      "Coarse list training of IVF builds: Lloyd iterations, restricted assignment (assignment_supers > 0: each row's\n"
      "nearest list among the members of its assignment_probes nearest supers) and the sample rows per list that\n"
      "build() trains on. The defaults are the exact recipe; indexes use automatic(lists) unless given one.")
      .def("__init__", [](Training* training, uint32_t iterations, uint32_t assignment_supers, uint32_t assignment_probes,
                          uint32_t sample_rows_per_list) {
        new (training) Training{iterations, assignment_supers, assignment_probes, sample_rows_per_list};
        training->validate();
      }, nb::kw_only(), "iterations"_a = exact.iterations, "assignment_supers"_a = exact.assignment_supers,
         "assignment_probes"_a = exact.assignment_probes, "sample_rows_per_list"_a = exact.sample_rows_per_list)
      .def_rw("iterations", &Training::iterations)
      .def_rw("assignment_supers", &Training::assignment_supers)
      .def_rw("assignment_probes", &Training::assignment_probes)
      .def_rw("sample_rows_per_list", &Training::sample_rows_per_list)
      .def_static("automatic", &Training::automatic, "lists"_a,
                  "The training an index with this many lists uses by default: fast from fast_lists lists on.")
      .def_ro_static("fast_lists", &Training::fast_lists)
      .def("__eq__", [](const Training& a, const Training& b) {
        return a.iterations == b.iterations && a.assignment_supers == b.assignment_supers &&
               a.assignment_probes == b.assignment_probes && a.sample_rows_per_list == b.sample_rows_per_list;
      })
      .def("__repr__", [](const Training& t) {
        return "TieredQCoarseTraining(iterations=" + std::to_string(t.iterations) + ", assignment_supers=" +
               std::to_string(t.assignment_supers) + ", assignment_probes=" + std::to_string(t.assignment_probes) +
               ", sample_rows_per_list=" + std::to_string(t.sample_rows_per_list) + ")";
      });

  using Router = dv::TieredQRouterConfig;
  const Router defaults{};
  nb::class_<Router>(module, "TieredQRouterConfig",
      "Two-level compressed coarse router settings (see the C++ notes); indexes with at least automatic_lists lists\n"
      "build it at these defaults unless router=False.")
      .def("__init__", [](Router* router, uint32_t lists_per_super, double bits_per_dimension, uint32_t rerank,
                          double expansion, uint32_t min_super_probes, uint64_t seed) {
        new (router) Router{lists_per_super, bits_per_dimension, rerank, expansion, min_super_probes, seed};
        router->validate();
      }, nb::kw_only(), "lists_per_super"_a = defaults.lists_per_super, "bits_per_dimension"_a = defaults.bits_per_dimension,
         "rerank"_a = defaults.rerank, "expansion"_a = defaults.expansion, "min_super_probes"_a = defaults.min_super_probes,
         "seed"_a = defaults.seed)
      .def_rw("lists_per_super", &Router::lists_per_super)
      .def_rw("bits_per_dimension", &Router::bits_per_dimension)
      .def_rw("rerank", &Router::rerank)
      .def_rw("expansion", &Router::expansion)
      .def_rw("min_super_probes", &Router::min_super_probes)
      .def_rw("seed", &Router::seed)
      .def_ro_static("automatic_lists", &Router::automatic_lists);
}
}  // namespace tieredq_python
