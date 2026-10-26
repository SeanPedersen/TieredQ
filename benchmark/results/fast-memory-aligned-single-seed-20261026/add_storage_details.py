"""Expose saved/reloaded allocation totals without changing serving kernels."""
from pathlib import Path
import difflib
p=Path('benchmark/native/module.cpp')
old=p.read_text()
needle='  size_t storage(const std::string& path) const {'
method='''  nb::dict storage_details(const std::string& path) const {
    tq_->save(path);
    const auto served = dv::TieredQIndex::load(path);
    nb::dict result;
    result["disk_bytes"] = std::filesystem::file_size(path);
    result["index_allocated_bytes"] = served.storage_bytes();
    result["model_bytes"] = served.codec().memory_bytes();
    result["index_shell_bytes"] = sizeof(dv::TieredQIndex) - sizeof(dv::TieredQ);
    result["resident_bytes"] = served.storage_bytes() + served.codec().memory_bytes()
        + sizeof(dv::TieredQIndex) - sizeof(dv::TieredQ);
    return result;
  }
'''
assert old.count(needle)==1
new=old.replace(needle,method+needle).replace('    .def("diagnostics", &Index::diagnostics)','    .def("storage_details", &Index::storage_details)\n    .def("diagnostics", &Index::diagnostics)')
p.write_text(new)
Path('storage-details.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='exact/benchmark/native/module.cpp',tofile='as-run/benchmark/native/module.cpp')))
