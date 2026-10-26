"""Experiment-only allocation introspection; changes access, never representation or kernels."""
from pathlib import Path
import sys
root=Path(sys.argv[1]);p=root/'benchmark/native/module.cpp';s=p.read_text()
s=s.replace('#include <tieredq/tieredq.h>','''#include <array>
#include <string>
#include <vector>
#include <utility>
#include <tieredq/detail/tieredq_scan_plan.h>
#define private public
#include <tieredq/tieredq.h>
#undef private
#include "../../src/tieredq_fast_storage.h"''')
pos=s.index('  void search(Matrix queries, Ids ids, int threads) const')
s=s[:pos]+'''  nb::dict allocations(const std::string& path) const {
    tq_->save(path);
    const auto served=dv::TieredQIndex::load(path);
    const auto& c=served.codec(); const auto& f=*served.fast_;
    nb::dict model, cache, result;
    auto bytes=[](const auto& v) { return v.capacity()*sizeof(typename std::decay_t<decltype(v)>::value_type); };
    model["object"]=sizeof(c); model["basis"]=bytes(c.basis_); model["centroids"]=bytes(c.centroids_);
    model["mean"]=bytes(c.mean_); model["sigma"]=bytes(c.sigma_); model["tiers"]=bytes(c.tiers_);
    model["half_norms"]=bytes(c.centroid_half_norms_); model["scan_plan"]=bytes(c.scan_plan_);
    size_t books=bytes(c.levels_);for (const auto& b:c.levels_) books+=bytes(b);
    model["codebooks"]=books;
    size_t known=sizeof(c)+bytes(c.basis_)+bytes(c.centroids_)+bytes(c.mean_)+bytes(c.sigma_)+bytes(c.tiers_)+bytes(c.centroid_half_norms_)+bytes(c.scan_plan_)+books;
    model["lazy_encoder_object"]=c.memory_bytes()-known;
    cache["object"]=sizeof(f);cache["levels_or_packed_codes"]=bytes(f.levels);
    cache["factor"]=bytes(f.factor);cache["correction"]=bytes(f.correction);cache["centroid_ids"]=bytes(f.centroid);
    cache["code_offsets"]=bytes(f.code_offset);cache["tier_of"]=bytes(f.tier_of);cache["level_scale"]=bytes(f.level_scale);
    size_t cb=bytes(f.codebooks);for (const auto& b:f.codebooks) cb+=bytes(b); cache["codebooks"]=cb;
    if (f.lookup) {
      const auto& l=*f.lookup;cache["lookup_object"]=sizeof(l);cache["lookup_projection"]=bytes(l.projection.panels);
      cache["lookup_groups"]=bytes(l.groups);cache["lookup_codes"]=bytes(l.codes);
      size_t lb=bytes(l.levels);for (const auto& b:l.levels) lb+=bytes(b);cache["lookup_codebooks"]=lb;
    }
    result["model"]=model;result["cache"]=cache;result["cache_total"]=served.storage_bytes();
    result["model_total"]=c.memory_bytes();result["serialized_bytes"]=std::filesystem::file_size(path);
    result["count"]=served.size();result["padded_count"]=f.factor.size();result["vector_bytes"]=served.bytes_per_vector();
    result["resident_bytes"]=served.storage_bytes()+c.memory_bytes();
    return result;
  }
''' +s[pos:]
s=s.replace('.def("diagnostics", &Index::diagnostics)', '.def("diagnostics", &Index::diagnostics)\n    .def("allocations", &Index::allocations)')
p.write_text(s)

method='''  bool verify_reload(const std::string& path, Matrix queries) const {
    tq_->save(path);
    const auto loaded=dv::TieredQIndex::load(path);
    const size_t count=queries.shape(0)*top_k;
    std::vector<float> before_scores(count),after_scores(count);
    std::vector<uint32_t> before_ids(count),after_ids(count);
    tq_->search_batch(queries.data(),queries.shape(0),top_k,before_scores.data(),before_ids.data());
    loaded.search_batch(queries.data(),queries.shape(0),top_k,after_scores.data(),after_ids.data());
    return before_scores==after_scores && before_ids==after_ids;
  }
'''

s=p.read_text().replace('  void search(Matrix queries, Ids ids, int threads) const',method+'  void search(Matrix queries, Ids ids, int threads) const').replace('.def("allocations", &Index::allocations)', '.def("allocations", &Index::allocations)\n    .def("verify_reload", &Index::verify_reload)')
p.write_text(s)
