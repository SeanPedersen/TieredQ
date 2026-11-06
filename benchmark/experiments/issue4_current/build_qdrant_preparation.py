"""Add a preprocessing-only diagnostic to a separate copy of the pinned Qdrant shim.

Requires an isolated Qdrant source checkout at the exact existing reference revision. The measured
shim and shared Python environment are never modified. Cargo runs offline with the existing lock.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys

w=Path(sys.argv[1]).resolve()
b=w/'source/benchmark'
q=w/'preparation-qdrant-source'
revision=subprocess.check_output(['git','-C',str(q),'rev-parse','HEAD'],text=True).strip()
assert revision=='222b2e44ab1c3f17e8246396ead6558e45041df7'
subprocess.run([sys.executable,str(b/'turboquant/patch_qdrant.py'),str(q)],check=True)
p=w/'preparation-qdrant'
p.mkdir(exist_ok=True)
shutil.copytree(b/'turboquant/native/src',p/'src',dirs_exist_ok=True)
shutil.copy2(b/'turboquant/native/Cargo.lock',p/'Cargo.lock')
cargo=(b/'turboquant/native/Cargo.toml').read_text().replace('../../../../qdrant',str(q))
(p/'Cargo.toml').write_text(cargo)
api=r'''
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tq_prepare_only(p:*const std::ffi::c_void,q:*const f32,n:usize,threads:usize) {
 let x=unsafe{&*p.cast::<Index>()};let q=unsafe{std::slice::from_raw_parts(q,n*x.dim)};
 static POOLS:std::sync::OnceLock<[rayon::ThreadPool;2]>=std::sync::OnceLock::new();
 let pools=POOLS.get_or_init(||[rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap(),rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap()]);
 if n==1 || threads==1 {for q in q.chunks_exact(x.dim){std::hint::black_box(x.quantizer.precompute_query(q));}return;}
 pools[1].install(||q.par_chunks(x.dim).for_each(|q|{std::hint::black_box(x.quantizer.precompute_query(q));}));
}
'''
src=p/'src/lib.rs'
src.write_text(src.read_text()+api)
(p/'preparation-api.rs').write_text(api)
env=__import__('os').environ | {'RUSTFLAGS':'-C target-cpu=native -C opt-level=3',
                              'CFLAGS':'-O3 -march=native -ffp-contract=off'}
cmd=['cargo','+1.99.1','build','--release','--locked','--offline','-j2']
subprocess.run(cmd,cwd=p,env=env,check=True)
(p/'build-provenance.json').write_text(json.dumps(dict(revision=revision,command=cmd,
    flags={key:env[key] for key in ['RUSTFLAGS','CFLAGS']},additional_api='preparation-api.rs'),indent=2))
