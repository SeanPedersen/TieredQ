// Coarse list options shared by the IVF and nested adapters: TieredQCoarseTraining and TieredQRouterConfig as Python
// classes, and the router argument (None: automatic for many lists; True/False; or a TieredQRouterConfig to build).
#pragma once
#include "arrays.h"
#include <optional>
#include <stdexcept>

namespace tieredq_python {
struct RouterChoice {
  std::optional<bool> use;  // unset: the index's automatic choice
  dv::TieredQRouterConfig config;
};
inline RouterChoice parse_router(nb::handle router) {
  if (router.is_none()) return {};
  if (nb::isinstance<dv::TieredQRouterConfig>(router)) return {true, nb::cast<dv::TieredQRouterConfig>(router)};
  if (PyBool_Check(router.ptr())) return {nb::cast<bool>(router), {}};
  throw std::invalid_argument("router must be None, a bool or a TieredQRouterConfig");
}
void bind_lists(nb::module_& module);
}  // namespace tieredq_python
