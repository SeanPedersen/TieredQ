"""Experiment-only like-for-like core serving allocations, separate from serialized bytes."""
from pathlib import Path
import sys
root=Path(sys.argv[1])
p=root/'benchmark/.deps/RaBitQ-Library/include/rabitqlib/utils/rotator.hpp';s=p.read_text();needle='    FhtKacRotator() = default;';assert s.count(needle)==1;s=s.replace(needle,'''    size_t allocated_bytes() const {
        // This rotation stores a free-function pointer in std::function (no heap target).
        if (!fht_float_.target<void(*)(float*)>()) throw std::runtime_error("Unexpected FHT callable allocation");
        return sizeof(*this)+flip_.capacity()*sizeof(uint8_t);
    }
'''+needle);p.write_text(s)
p=root/'benchmark/native/rabitq_flat.h';s=p.read_text();needle='  void save(const std::string& path) const {';assert s.count(needle)==1;s=s.replace(needle,'''  std::array<size_t,5> serving_allocations() const {
    const auto* rotation=dynamic_cast<const rabitqlib::rotator_impl::FhtKacRotator*>(rotator_.get());
    if (!rotation) throw std::runtime_error("Unexpected rotation type");
    return {sizeof(*this),rotation->allocated_bytes(),center_.capacity()*sizeof(float),binary_.allocated_bytes(),extra_.allocated_bytes()};
  }
'''+needle);p.write_text(s)
p=root/'benchmark/native/module.cpp';s=p.read_text();needle='  void search(Matrix queries, Ids ids, int threads) const';assert s.count(needle)==1;s=s.replace(needle,'''  nb::dict memory_metrics(const std::string& path) const {
    nb::dict result, components;
    size_t ram=0;
    if (rq_) {
      rq_->save(path);
      const auto a=rq_->serving_allocations();
      const char* names[]={"core_object","rotation_object_and_flips","center","binary_codes_and_factors","extra_codes_and_factors"};
      for (size_t i=0;i<a.size();++i) {components[names[i]]=a[i];ram+=a[i];}
    } else {
      tq_->save(path);
      const auto loaded=dv::TieredQIndex::load(path);
      components["model"]=loaded.codec().memory_bytes(); components["cache"]=loaded.storage_bytes();
      components["index_object_excluding_embedded_codec"]=sizeof(loaded)-sizeof(loaded.codec());
      ram=loaded.codec().memory_bytes()+loaded.storage_bytes()+sizeof(loaded)-sizeof(loaded.codec());
    }
    result["ram_components"]=components; result["ram_bytes"]=ram;
    result["disk_bytes"]=std::filesystem::file_size(path); result["count"]=count_;
    result["disk_bits_per_vector"]=double(std::filesystem::file_size(path))*8/count_;
    result["ram_bits_per_vector"]=double(ram)*8/count_;
    return result;
  }
'''+needle).replace('.def("diagnostics", &Index::diagnostics)', '.def("diagnostics", &Index::diagnostics)\n    .def("memory_metrics", &Index::memory_metrics)');p.write_text(s)
