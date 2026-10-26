//! Minimal C ABI: production TQ+ fit/encode and asymmetric exhaustive compressed scoring.
use std::{ffi::{CStr,c_char}, path::Path, sync::atomic::AtomicBool};
use quantization::{ encoded_storage::TestEncodedStorageBuilder, encoded_vectors_tq::{EncodedVectorsTQ,Metadata,new_turbo_quantizer_from_metadata,get_quantized_vector_size}, VectorParameters,DistanceType,turboquant::{TQBits,TQMode,TQRotation,quantization::TurboQuantizer}};
use rayon::prelude::*;
struct Index { quantizer:TurboQuantizer, codes:Vec<u8>, stride:usize, dim:usize, encode_s:f64 }
fn text<'a>(p:*const c_char)-> &'a str {unsafe {CStr::from_ptr(p)}.to_str().unwrap()}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tq_build(data:*const f32,n:usize,d:usize,bits:u32,path:*const c_char)->*mut std::ffi::c_void {
 let bits=match bits {1=>TQBits::Bits1,2=>TQBits::Bits2,4=>TQBits::Bits4,_=>panic!("bits")};
 let path=Path::new(text(path));std::fs::create_dir_all(path).unwrap();
 let vp=VectorParameters{dim:d,distance_type:DistanceType::Cosine,invert:false,deprecated_count:None};
 let stride=get_quantized_vector_size(&vp,bits,TQMode::Plus);
 let values=unsafe{std::slice::from_raw_parts(data,n*d)};
 let encoded=EncodedVectorsTQ::encode(values.chunks_exact(d),std::iter::repeat_n(true,n),TestEncodedStorageBuilder::new(None,stride),&vp,bits,TQMode::Plus,TQRotation::Padded,false,8,Some(&path.join("metadata.json")),&AtomicBool::new(false)).unwrap();
 let codes:Vec<u8>=(0..n).flat_map(|i|encoded.get_quantized_vector(i as u32).into_owned()).collect();
 let codes=codes.into_boxed_slice().into_vec();
 std::fs::write(path.join("codes.bin"),&codes).unwrap();
 let quantizer=new_turbo_quantizer_from_metadata(encoded.get_metadata()).unwrap();
 Box::into_raw(Box::new(Index{quantizer,codes,stride,dim:d,encode_s:encoded.benchmark_encode_seconds})).cast()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tq_load(path:*const c_char)->*mut std::ffi::c_void {
 let path=Path::new(text(path)); let meta:Metadata=serde_json::from_slice(&std::fs::read(path.join("metadata.json")).unwrap()).unwrap();
 let stride=get_quantized_vector_size(&meta.vector_parameters,meta.bits,meta.mode);
 let codes=std::fs::read(path.join("codes.bin")).unwrap();assert_eq!(codes.len()%stride,0);
 let quantizer=new_turbo_quantizer_from_metadata(&meta).unwrap();
 Box::into_raw(Box::new(Index{quantizer,codes,stride,dim:meta.vector_parameters.dim,encode_s:0.0})).cast()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tq_stats(p:*const std::ffi::c_void,out:*mut f64) {
 let x=unsafe{&*p.cast::<Index>()};let out=unsafe{std::slice::from_raw_parts_mut(out,5)};
 out.copy_from_slice(&[x.codes.capacity() as f64,x.quantizer.heap_size_bytes() as f64,size_of::<Index>() as f64,x.encode_s,x.stride as f64]);
}
fn search_one(x:&Index,q:&[f32],out:&mut [u32]) {
 let query=x.quantizer.precompute_query(q);let n=x.codes.len()/x.stride;let mut scores=vec![0.0;n];
 x.quantizer.score_precomputed_batch(&query,&x.codes,x.stride,&mut scores);
 let mut ids:Vec<usize>=(0..n).collect();let cmp=|a:&usize,b:&usize|scores[*b].total_cmp(&scores[*a]).then(a.cmp(b));
 assert!(scores.iter().all(|s|s.is_finite()));ids.select_nth_unstable_by(10,cmp);ids[..10].sort_unstable_by(cmp);
 for (a,b) in out.iter_mut().zip(ids) {*a=b as u32;}
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tq_search(p:*const std::ffi::c_void,q:*const f32,n:usize,out:*mut u32,threads:usize) {
 let x=unsafe{&*p.cast::<Index>()};let q=unsafe{std::slice::from_raw_parts(q,n*x.dim)};let out=unsafe{std::slice::from_raw_parts_mut(out,n*10)};
 // Persistent pools avoid per-call thread creation; pool allocations are reported as runtime state.
 static POOLS:std::sync::OnceLock<[rayon::ThreadPool;2]>=std::sync::OnceLock::new();
 let pools=POOLS.get_or_init(||[rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap(),rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap()]);
 if n==1 || threads==1 {for (q,o) in q.chunks_exact(x.dim).zip(out.chunks_exact_mut(10)){search_one(x,q,o);}return;}
 pools[1].install(||out.par_chunks_mut(10).zip(q.par_chunks(x.dim)).for_each(|(o,q)|search_one(x,q,o)));
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tq_free(p:*mut std::ffi::c_void) {drop(unsafe{Box::from_raw(p.cast::<Index>())});}

/// Compare upstream per-vector and contiguous-batch scoring, independently of top-k.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tq_validate(p:*const std::ffi::c_void,q:*const f32)->bool {
 let x=unsafe{&*p.cast::<Index>()};let q=unsafe{std::slice::from_raw_parts(q,x.dim)};
 let query=x.quantizer.precompute_query(q);let mut batch=vec![0.0;x.codes.len()/x.stride];
 x.quantizer.score_precomputed_batch(&query,&x.codes,x.stride,&mut batch);
 batch.iter().zip(x.codes.chunks_exact(x.stride)).all(|(a,b)|a.is_finite() && a.to_bits()==x.quantizer.score_precomputed(&query,b).to_bits())
}
