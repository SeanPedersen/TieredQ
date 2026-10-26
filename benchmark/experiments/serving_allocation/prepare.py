"""Patch a fresh isolated snapshot with reversible experimental allocator and byte lookup variants."""
from pathlib import Path
import argparse


def edit(root, name, old, new):
    path = root / name
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f'changed patch context: {name}: {old[:80]}')
    path.write_text(text.replace(old, new))


def prepare(root):
    edit(root, 'src/tieredq.cpp', '  auto gain=[&](int rank) { unsigned b=bits[rank]; return std::max(0.0,eigenvalues[d-1-rank])*(error[b]-error[b+1]); };', '''  const char* allocation_env=std::getenv("TIEREDQ_EXPERIMENT_ALLOCATION");
  const std::string allocation=allocation_env ? allocation_env : "marginal";
  if (allocation!="marginal" && allocation!="log" && allocation!="cap4") bad("unknown experiment allocation");
  if (allocation=="cap4" && config.code_bits<=4*config.dimensions) config.max_bits=4;
  auto gain=[&](int rank) {
    unsigned b=bits[rank];
    const double variance=std::max(0.0,eigenvalues[d-1-rank]);
    if (allocation=="log") return .5*std::log2(std::max(variance,1e-300))-b;
    return variance*(error[b]-error[b+1]);
  };''')
    edit(root, 'src/tieredq_scan_lookup_x86.h', '  std::vector<int8_t> tables;', '  std::vector<int8_t> tables;\n  std::vector<int32_t> byte_tables;')
    edit(root, 'src/tieredq_scan_lookup_x86.h', '  return query;\n}', '''  const char* grouping=std::getenv("TIEREDQ_EXPERIMENT_GROUPING");
  if (grouping && std::string(grouping)=="byte") {
    constexpr size_t byte_entries=256;
    query.byte_tables.resize(cache.pairs()*byte_entries);
    for (size_t pair=0;pair<cache.pairs();++pair) for (size_t code=0;code<byte_entries;++code)
      query.byte_tables[pair*byte_entries+code]=int(query.tables[2*pair*lookup_table_entries+(code&15)])
          +int(query.tables[(2*pair+1)*lookup_table_entries+(code>>4)]);
  }
  return query;
}''')
    path=root/'src/tieredq_scan_lookup_avx512.h'
    text=path.read_text()
    point='// first is aligned to lookup_block_docs; count may end within the final block.'
    kernel='''// Same stored bytes, integer sums, scales, and FP32 accumulation order as the register path.
__attribute__((target("avx512f,avx512bw")))
inline void scan_byte_lookup_avx512_range(const FastScanCache& c,const FastScanLookupQuery& q,
    size_t first,size_t count,float* out) {
  const auto& lookup=*c.lookup;
  constexpr size_t byte_entries=256;
  for (size_t doc=first;doc<first+count;doc+=lookup_vector_docs) {
    auto dot=_mm512_setzero_ps();
    const auto* base=lookup.codes.data()+doc/lookup_block_docs*lookup.pairs()*lookup_block_docs+doc%lookup_block_docs;
    for (size_t start=0;start<lookup.groups.size();start+=lookup_scale_groups) {
      auto sum=_mm512_setzero_si512();
      const size_t end=std::min(start+lookup_scale_groups,lookup.groups.size());
      for (size_t group=start;group<end;group+=lookup_pair_groups) {
        const auto code=_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(base+group/lookup_pair_groups*lookup_block_docs)));
        sum=_mm512_add_epi32(sum,_mm512_i32gather_epi32(code,q.byte_tables.data()+group/lookup_pair_groups*byte_entries,sizeof(int32_t)));
      }
      dot=_mm512_add_ps(dot,_mm512_mul_ps(_mm512_cvtepi32_ps(sum),_mm512_set1_ps(q.scales[start/lookup_scale_groups])));
    }
    lookup_epilogue(c,q,doc,dot,out,first,first+count);
  }
}
'''
    if text.count(point)!=1:
        raise ValueError('changed kernel context')
    path.write_text(text.replace(point,kernel+point))
    edit(root, 'src/tieredq_scan_lookup_avx512.h', '  const auto& lookup=*c.lookup;\n  constexpr size_t chunks', '''  if (!q.byte_tables.empty()) { scan_byte_lookup_avx512_range(c,q,first,count,out); return; }
  const auto& lookup=*c.lookup;
  constexpr size_t chunks''')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    prepare(parser.parse_args().root)
