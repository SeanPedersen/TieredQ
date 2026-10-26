"""Replace only packed-code register lookup with 256-entry FP32 gather tables in a separate build."""
from pathlib import Path
import sys

root=Path(sys.argv[1])
p=root/'src/tieredq_scan_codes_x86.h'
s=p.read_text()
s=s.replace('std::vector<float> table(code_table_pairs(width)*code_pair_entries);','std::vector<float> table(256);')
start=s.index('inline __m512 table_lookup_avx512(')
end=s.index('// Codes at a runtime bit offset:',start)
s=s[:start]+'''inline __m512 table_lookup_avx512(const float* table,__m512i code) {
  constexpr unsigned byte_mask=255;
  return _mm512_i32gather_ps(_mm512_and_si512(code,_mm512_set1_epi32(byte_mask)),table,sizeof(float));
}
'''+s[end:]
s=s.replace('walk_period_avx512(const __m512* table','walk_period_avx512(const float* table')
old='''  __m512 table[2*pairs];
  for (size_t i=0;i<2*pairs;++i) table[i]=_mm512_loadu_ps(segment.table.data()+i*code_register_lanes);'''
assert s.count(old)==1
s=s.replace(old,'  const float* table=segment.table.data();')
p.write_text(s)
