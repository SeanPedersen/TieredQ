"""Build a separate diagnostic adapter exposing query preparation without scanning.

Reads the measured build's objects; writes only preparation-build/. Never rewrites a loaded binary.
"""
import difflib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

w=Path(sys.argv[1]).resolve()
s=w/'source'
b=s/'benchmark/build-native'
p=w/'preparation-build'
p.mkdir(exist_ok=True)
patches=[]
def altered(source,destination,transform):
    old=source.read_text();new=transform(old)
    assert old!=new,source
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(new)
    patches.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                                       fromfile=str(source.relative_to(s)),tofile=str(destination.relative_to(p))))

altered(s/'include/tieredq/tieredq.h',p/'include/tieredq/tieredq.h',lambda t:t.replace(
    'class TieredQIndex {\n public:',
    'class TieredQIndex {\n public:\n  void issue4_prepare(const float* queries,size_t rows) const;'))
method=r'''
namespace tq {
void TieredQIndex::issue4_prepare(const float* queries,size_t rows) const {
  const ModelQueries model(codec_,queries,rows);
  std::vector<TieredQQuery> weights;
  if (fine_tiers_) weights=fine_query_weights(model.data,rows);
#ifdef TQ_X86
  if (rows==1 && tieredq_detail::fast_scan_backend(*fast_)==tieredq_detail::FastScanBackend::avx512 &&
      tieredq_detail::fast_range_checkpoint_supported(*fast_,scan_codec().tiers())) {
    const auto prepared=fine_tiers_ ? weights : scan_codec().model_query_weights(model.data,1);
    const auto bounds=tieredq_detail::prepare_fast_range_bounds(*fast_,scan_codec().tiers(),prepared[0]);
    asm volatile("" : : "g"(&prepared),"g"(&bounds) : "memory");
    return;
  }
#endif
  tieredq_detail::with_fast_scan(*fast_,scan_codec(),model.data,rows,[&](const auto& prepared,size_t,auto) {
    if (rows==1 && tieredq_detail::lookup_range_scan_supported(*fast_,scan_codec().tiers())) {
      const auto bounds=tieredq_detail::prepare_lookup_range_bounds(*fast_,scan_codec().tiers(),prepared[0].lookup);
      asm volatile("" : : "g"(&bounds) : "memory");
    }
    asm volatile("" : : "g"(&prepared) : "memory");
  },fine_tiers_ ? weights.data() : nullptr);
}
}
'''
# The callback is instantiated for code-lookup and scalar query types as well.
method=method.replace('if (rows==1 && tieredq_detail::lookup_range_scan_supported',
    'if constexpr (requires { prepared[0].lookup; }) if (rows==1 && tieredq_detail::lookup_range_scan_supported')
altered(s/'src/tieredq_index.cpp',p/'tieredq_index.cpp',lambda t:t+method)

def rabitq(t):
    start=t.index('  void search(const float* vector,size_t k,uint32_t* ids) const {')
    end=t.index('    for (size_t first=0;first<count_;first+=batch_size)',start)
    prefix=t[start:end].replace('void search(const float* vector,size_t k,uint32_t* ids)',
                                'void prepare_only(const float* vector)').replace('top(k);','top(10);')
    return t[:start]+prefix+'    asm volatile("" : : "g"(&query) : "memory");\n  }\n'+t[start:]
altered(s/'benchmark/native/rabitq_flat.h',p/'rabitq_flat.h',rabitq)

def module(t):
    t=re.sub(r'\bIndex\b','PreparationIndex',t)
    t=re.sub(r'\bNestedIndex\b','PreparationNestedIndex',t)
    t=t.replace('NB_MODULE(codec_native, m)','NB_MODULE(codec_preparation, m)')
    marker='  nb::dict diagnostics() const {'
    api='''  void prepare_only(Matrix queries,int threads) const {
    omp_set_num_threads(threads);
    if (tq_) { tq_->issue4_prepare(queries.data(),queries.shape(0)); return; }
    #pragma omp parallel for schedule(static) if(queries.shape(0)>1)
    for (size_t i=0;i<queries.shape(0);++i) rq_->prepare_only(queries.data()+i*width_);
  }
'''
    t=t.replace(marker,api+marker)
    t=t.replace('.def("diagnostics", &PreparationIndex::diagnostics)',
                '.def("prepare_only", &PreparationIndex::prepare_only)\n    .def("diagnostics", &PreparationIndex::diagnostics)')
    return t
altered(s/'benchmark/native/module.cpp',p/'module.cpp',module)
commands=json.loads((b/'compile_commands.json').read_text())
for basename in ('tieredq_index.cpp','module.cpp'):
    entry=next(row for row in commands if Path(row['file']).name==basename)
    cmd=shlex.split(entry['command'])
    cmd[cmd.index('-o')+1]=str(p/(basename+'.o'))
    cmd[cmd.index('-c')+1]=str(p/basename)
    cmd[1:1]=['-I'+str(p/'include'),'-I'+str(s/'src')]
    subprocess.run(cmd,cwd=entry['directory'],check=True)
link=shlex.split((b/'CMakeFiles/codec_native.dir/link.txt').read_text())
link=[arg for arg in link if not arg.startswith('-Wl,--dependency-file=')]
name=link[link.index('-o')+1].replace('codec_native','codec_preparation')
link[link.index('-o')+1]=str(p/name)
link[link.index('CMakeFiles/codec_native.dir/module.cpp.o')]=str(p/'module.cpp.o')
link.insert(link.index('tieredq/libtieredq.a'),str(p/'tieredq_index.cpp.o'))
subprocess.run(link,cwd=b,check=True)
(p/'diagnostic.patch').write_text(''.join(patches))
(p/'link-command.json').write_text(json.dumps(link,indent=2))
print('Separate preprocessing adapter ready',flush=True)
