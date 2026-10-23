// Optional Python module; the codec itself has no Python dependencies.
#include "arrays.h"
NB_MODULE(_core, module) {
  module.doc() = "Trained TieredQ exhaustive in-memory index";
  tieredq_python::bind_index(module);
}
