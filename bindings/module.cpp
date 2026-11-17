// Optional Python module; the codec itself has no Python dependencies.
#include "arrays.h"
#include "lists.h"
NB_MODULE(_core, module) {
  module.doc() = "Trained TieredQ flat, IVF and nested indexes, and the unquantized IVF-Flat";
  tieredq_python::bind_index(module);
  tieredq_python::bind_lists(module);
  tieredq_python::bind_ivf(module);
  tieredq_python::bind_ivf_flat(module);
  tieredq_python::bind_nested(module);
}
