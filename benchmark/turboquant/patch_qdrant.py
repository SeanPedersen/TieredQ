"""Seed the production calibration sampler and expose serving allocation/timing observations."""
from pathlib import Path
import sys
root=Path(sys.argv[1])
def change(name,old,new):
 p=root/name;s=p.read_text();assert s.count(old)==1,(name,s.count(old));p.write_text(s.replace(old,new))
change('lib/quantization/src/quantile.rs','rand::seq::index::sample(&mut rand::rng(), count, actual_sample)', 'rand::seq::index::sample(&mut <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0), count, actual_sample)')
change('lib/quantization/src/turboquant/rotation.rs','    pub fn dim(&self) -> usize {','    pub fn benchmark_heap_bytes(&self) -> usize {\n        self.forward_maps.iter().chain(self.backward_maps.iter()).map(|v| v.capacity()*size_of::<u32>()).sum()\n    }\n\n    pub fn dim(&self) -> usize {')
change('lib/quantization/src/turboquant/quantization.rs','    pub(crate) fn heap_size_bytes(&self) -> usize {','    pub fn heap_size_bytes(&self) -> usize {')
change('lib/quantization/src/turboquant/quantization.rs','        error_correction.as_ref().map_or(0, |ec| {','        self.rotation.benchmark_heap_bytes() + error_correction.as_ref().map_or(0, |ec| {')
p=root/'lib/quantization/src/encoded_vectors_tq.rs'
s=p.read_text()
s=s.replace('    encoding_buffer: Vec<f64>,', '    encoding_buffer: Vec<f64>,\n    pub benchmark_encode_seconds: f64,',1)
s=s.replace('        for vector in data {','        let benchmark_encode_start = std::time::Instant::now();\n        for vector in data {',1)
s=s.replace('        let encoded_vectors = storage_builder','        let benchmark_encode_seconds = benchmark_encode_start.elapsed().as_secs_f64();\n        let encoded_vectors = storage_builder',1)
s=s.replace('            metadata_path: meta_path.map(PathBuf::from),','            metadata_path: meta_path.map(PathBuf::from),\n            benchmark_encode_seconds,',1)
s=s.replace('            metadata_path: Some(meta_path.to_path_buf()),','            metadata_path: Some(meta_path.to_path_buf()),\n            benchmark_encode_seconds: 0.0,',1)
s=s.replace('            metadata_path: _,','            metadata_path: _,\n            benchmark_encode_seconds: _,',1)
p.write_text(s)
