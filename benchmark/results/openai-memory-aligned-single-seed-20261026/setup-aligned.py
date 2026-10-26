"""Create an isolated committed-source workspace with an exact-budget binding."""
from pathlib import Path
import subprocess,hashlib,json,difflib
r=Path('/home/sean/experiments/TieredQ-openai-aligned-20261026-akXzMJ')
archive=Path('/home/sean/experiments/TieredQ-issue4-current-20261026-tpj3ze/committed-source.tar')
subprocess.run(['tar','xf',str(archive),'-C',str(r)],check=True)
(r/'SOURCE_REVISION').write_text('3c9dbf590209dabf8ecccd3385fd56254779c70d\n')
b=r/'benchmark'; (b/'data').symlink_to('/home/sean/experiments/TieredQ-issue4-20261023/TieredQ/benchmark/data',target_is_directory=True)
(b/'.deps').mkdir();subprocess.run(['cp','-a','/home/sean/experiments/TieredQ-issue12-recall/benchmark/.deps/RaBitQ-Library',str(b/'.deps')],check=True)
p=b/'native/module.cpp';old=p.read_text();new=old.replace('bool l2, bool fast = true)','bool l2, bool fast = true, uint32_t exact_code_bits = 0)')
new=new.replace('bits * (width_ - (l2 ? 1 : 0))','exact_code_bits ? exact_code_bits : bits * (width_ - (l2 ? 1 : 0))')
needle='    .def("diagnostics", &Index::diagnostics)'
new=new.replace(needle,'    .def(nb::init<Matrix,size_t,uint64_t,bool,bool,bool,uint32_t>(), nb::call_guard<nb::gil_scoped_release>())\n'+needle)
assert new!=old;p.write_text(new)
(r/'exact-budget-binding.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='committed/benchmark/native/module.cpp',tofile='as-run/benchmark/native/module.cpp')))
(r/'source-sha256.json').write_text(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','include','benchmark/native'] for p in (r/folder).rglob('*') if p.is_file()},indent=2))
print('Prepared exact-budget binding; codec sources unchanged',flush=True)
