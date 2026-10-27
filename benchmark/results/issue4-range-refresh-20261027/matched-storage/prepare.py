"""Create an isolated adapter exposing exact code budgets; preserve measured kernels."""
from pathlib import Path
import hashlib,json,shutil
root=Path('/home/sean/experiments/TieredQ-issue4-range-refresh-20261027-Rkn542')
out=root/'matched-storage'; out.mkdir(exist_ok=True)
native=out/'native';native.mkdir(exist_ok=True)
original=(root/'source/benchmark/native/module.cpp').read_text()
patched=original.replace('bool fast = true)', 'bool fast = true, size_t code_bits = 0)').replace('bits * (width_ - (l2 ? 1 : 0))','code_bits ? code_bits : bits * (width_ - (l2 ? 1 : 0))').replace('.def("diagnostics", &Index::diagnostics)', '.def(nb::init<Matrix,size_t,uint64_t,bool,bool,bool,size_t>(), nb::call_guard<nb::gil_scoped_release>())\n    .def("diagnostics", &Index::diagnostics)')
assert patched!=original
(native/'module.cpp').write_text(patched)
for name in ['rabitq_flat.h','reference.cpp']: shutil.copy2(root/'source/benchmark/native'/name,native/name)
cmake=(root/'source/benchmark/native/CMakeLists.txt').read_text().replace('add_subdirectory(../.. tieredq)',f'add_subdirectory({root}/source tieredq)')
(native/'CMakeLists.txt').write_text(cmake)
(out/'adapter.patch').write_text(__import__('subprocess').run(['diff','-u',str(root/'source/benchmark/native/module.cpp'),str(native/'module.cpp')],capture_output=True,text=True).stdout)
print('Prepared isolated adapter',out,flush=True)
