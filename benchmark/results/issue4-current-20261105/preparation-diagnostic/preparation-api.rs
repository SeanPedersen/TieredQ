
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tq_prepare_only(p:*const std::ffi::c_void,q:*const f32,n:usize,threads:usize) {
 let x=unsafe{&*p.cast::<Index>()};let q=unsafe{std::slice::from_raw_parts(q,n*x.dim)};
 static POOLS:std::sync::OnceLock<[rayon::ThreadPool;2]>=std::sync::OnceLock::new();
 let pools=POOLS.get_or_init(||[rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap(),rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap()]);
 if n==1 || threads==1 {for q in q.chunks_exact(x.dim){std::hint::black_box(x.quantizer.precompute_query(q));}return;}
 pools[1].install(||q.par_chunks(x.dim).for_each(|q|{std::hint::black_box(x.quantizer.precompute_query(q));}));
}
