// Independent persisted-model references for layout, binary16, reconstruction, and scans.
#include "tieredq_basis_panels.h"
#include "tieredq/tieredq.h"
#include "tieredq_layout.h"
#include "tieredq_encode.h"
#include "tieredq_train_linalg.h"
#include "tieredq_train_kmeans.h"
#include "tieredq_train_stats.h"
#include "tieredq_train_covariance.h"
#include "tieredq_train_residuals.h"
#include "tieredq_train_codebook.h"
#include "tieredq_train_codebook_data.h"
#include "tieredq_scan_groups.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace dv;
namespace {
void require(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F run) { bool caught=false; try { run(); } catch (const std::exception&) { caught=true; } require(caught,"expected rejection"); }
struct Bytes {
  std::vector<uint8_t> data;
  void u32(uint32_t value) { for (unsigned i=0;i<4;++i) data.push_back(uint8_t(value>>(i*8))); }
  void u64(uint64_t value) { u32(uint32_t(value)); u32(uint32_t(value>>32)); }
  void floats(const std::vector<float>& values) { u64(values.size()); for (float value:values) u32(std::bit_cast<uint32_t>(value)); }
};
uint64_t hash(const std::vector<uint8_t>& values) { uint64_t h=14695981039346656037ULL; for (auto v:values) { h^=v; h*=1099511628211ULL; } return h; }
void persist(const std::string& path,const std::vector<uint8_t>& model,const std::vector<uint8_t>& payload={},size_t count=0,bool index=false) {
  Bytes header; header.data={'T','I','E','R','E','D','Q',0};
  header.u32(1); header.u32(index ? 2 : 1); header.u64(80); header.u64(model.size());
  header.u64(80+model.size()); header.u64(payload.size()); header.u64(count); header.u64(hash(model)); header.u64(hash(payload)); header.u64(0);
  std::ofstream out(path,std::ios::binary);
  for (const auto* part:std::array<const std::vector<uint8_t>*,3>{&header.data,&model,&payload}) out.write(reinterpret_cast<const char*>(part->data()),part->size());
}
std::vector<uint8_t> read_bytes(const std::string& path) { std::ifstream in(path,std::ios::binary); return {std::istreambuf_iterator<char>(in),{}}; }
// No PCA dependency: identity basis and a nonzero centroid.
Bytes model(unsigned d,unsigned bits) {
  Bytes out; for (auto value:{d,d*bits,1u,0u,2u,(d*bits+7)/8,(d*bits+7)/8+5u,1u,0u,d,bits,0u,0u}) out.u32(value);
  std::vector<float> center(d,0),eye(size_t(d)*d,0),zero(d,0),sigma(d,1); center[0]=2;
  for (unsigned j=0;j<d;++j) eye[size_t(j)*d+j]=1;
  out.floats(center);
  out.floats(eye); out.floats(zero); out.floats(sigma);
  out.u32(9);
  for (unsigned b=0;b<=8;++b) { std::vector<float> levels(b ? 1u<<b : 0); for (size_t i=0;i<levels.size();++i) levels[i]=float(i)-float(levels.size()-1)/2; out.floats(levels); }
  return out;
}
TieredQPanels encoder_panels(const std::vector<float>& basis,size_t d,size_t m,bool transpose=false) {
  TieredQPanels panels; panels.depth=transpose ? m : d; panels.columns=transpose ? d : m;
  panels.data.resize(panels.depth*panels.padded_columns());
  constexpr size_t tile=TieredQPanels::PANEL;
  for (size_t k=0;k<panels.depth;++k) for (size_t c=0;c<panels.columns;++c)
    panels.data[(c/tile*panels.depth+k)*tile+c%tile]=transpose ? basis[c*m+k] : basis[k*m+c];
  return panels;
}
void centroid_update_parity() {
  constexpr size_t rows=513;
  for (size_t dimensions:{size_t(67),size_t(1537)}) {
    std::mt19937 rng(211); std::normal_distribution<float> normal;
    std::vector<float> sample(rows*dimensions);
    for (auto& value:sample) value=normal(rng);
    for (size_t clusters:{size_t(1),size_t(7),size_t(256)}) {
      std::vector<uint32_t> ids(rows);
      for (size_t i=0;i<rows;++i) ids[i]=uint32_t(rng()%clusters);
      if (clusters>1) for (auto& id:ids) if (id==clusters-1) id=0;
      std::vector<float> reference(clusters*dimensions,3),actual=reference;
      tieredq_detail::update_centroids_reference(sample.data(),rows,dimensions,ids.data(),reference);
      tieredq_detail::update_centroids(sample.data(),rows,dimensions,ids.data(),actual);
      for (size_t i=0;i<actual.size();++i)
        require(std::bit_cast<uint32_t>(actual[i])==std::bit_cast<uint32_t>(reference[i]),"centroid update order/empty cluster");
    }
  }
}
void training_covariance_layout() {
  using tieredq_detail::TrainingLinalg;
  const auto& backend=TrainingLinalg::get();
  if (!backend.syrk) return;
  constexpr int dimensions=3,rows=5;
  const std::vector<double> residual{1,7,2, 5,-3,8, -4,2,9, 6,1,-2, 3,5,4},mean{2,1,3};
  tieredq_detail::StageProfile profile;
  const auto actual=backend.covariance(residual,mean,dimensions,rows,profile);
  for (int i=0;i<dimensions;++i) for (int j=0;j<=i;++j) {
    double sum=0;
    for (int r=0;r<rows;++r) sum+=(residual[r*dimensions+i]-mean[i])*(residual[r*dimensions+j]-mean[j]);
    require(std::abs(actual[j*dimensions+i]-sum/(rows-1))<1e-12,"BLAS covariance lower-triangle layout");
  }
}
void wide_grid_threshold_parity() {
  using namespace tieredq_detail;
  if (!TieredQ::kernel_available(TieredQKernel::avx512)) return;
  const std::vector<float> center{0},basis{1},mean{0},sigma{1};
  for (unsigned bits=5;bits<=8;++bits) {
    std::vector<std::vector<float>> levels(9);
    for (unsigned i=0;i<(1u<<bits);++i) levels[bits].push_back(float(i/3)-40);
    const std::vector<TieredQTier> tiers{{0,1,bits,0,0}};
    const EncodingModel model{1,center,mean,sigma,levels,tiers};
    EncodingWorkspace work(model); work.v[0]=1;
    for (double threshold:work.thresholds[bits]) for (double value:{std::nextafter(threshold,-INFINITY),threshold,std::nextafter(threshold,INFINITY)}) {
      work.z[0]=value;
      std::array<double,scale_grid_storage> expected_dots{},expected_norms{},dots{},norms{};
      scale_grid_terms(model,work,expected_dots.data(),expected_norms.data(),TieredQKernel::scalar);
      scale_grid_terms(model,work,dots.data(),norms.data(),TieredQKernel::avx512);
      for (size_t point=0;point<scale_grid_points;++point) {
        require(std::bit_cast<uint64_t>(dots[point])==std::bit_cast<uint64_t>(expected_dots[point]),"wide grid dot/threshold parity");
        require(std::bit_cast<uint64_t>(norms[point])==std::bit_cast<uint64_t>(expected_norms[point]),"wide grid norm/threshold parity");
      }
    }
  }
}
void encoder_boundaries() {
  {
    const std::vector<double> eigenvalues{4./3,16./3},residual_mean{2,3};
    const std::vector<int> ranks{1,0};
    const std::vector<TieredQTier> tiers{{0,2,2,0,0}};
    const std::vector<std::vector<float>> rotations{{1,0,0,1}};
    const std::vector<float> basis{1,0,0,1}; std::vector<float> mean,sigma;
    tieredq_detail::spectral_projection_stats(eigenvalues,ranks,tiers,rotations,residual_mean,basis,4,mean,sigma);
    require(mean==std::vector<float>({2,3}) && sigma==std::vector<float>({1,2}),"spectral population moments");
  }
  {
    constexpr size_t rows=131,dimensions=67,clusters=7;
    std::mt19937 rng(311); std::normal_distribution<float> normal;
    std::vector<float> sample(rows*dimensions),centroids(clusters*dimensions);
    for (auto& value:sample) value=normal(rng);
    for (auto& value:centroids) value=normal(rng);
    std::vector<uint32_t> ids(rows);
    for (auto& id:ids) id=rng()%clusters;
    std::vector<double> residual(rows*dimensions),mean(dimensions);
    for (size_t i=0;i<rows;++i) for (size_t j=0;j<dimensions;++j) {
      residual[i*dimensions+j]=double(sample[i*dimensions+j])-centroids[ids[i]*dimensions+j];
      mean[j]+=residual[i*dimensions+j];
    }
    for (auto& value:mean) value/=rows;
    const auto actual=tieredq_detail::centered_residuals(sample.data(),rows,dimensions,ids.data(),centroids);
    for (size_t i=0;i<rows;++i) for (size_t j=0;j<dimensions;++j)
      require(std::bit_cast<uint64_t>(actual.values[i*dimensions+j])==std::bit_cast<uint64_t>(residual[i*dimensions+j]-mean[j]),"centered residual order/parity");
  }
#if defined(__x86_64__) || defined(__i386__)
  if (TieredQ::kernel_available(TieredQKernel::avx512)) {
    constexpr size_t rows=131;
    for (size_t dimensions:{size_t(1),size_t(7),size_t(17),size_t(200)}) {
      std::mt19937 rng(723); std::normal_distribution<double> normal;
      std::vector<double> centered(rows*dimensions);
      for (auto& value:centered) value=normal(rng);
      const auto reference=tieredq_detail::small_covariance_reference(centered,dimensions,rows);
      const auto actual=tieredq_detail::small_covariance_avx512(centered,dimensions,rows);
      for (size_t i=0;i<actual.size();++i)
        require(std::bit_cast<uint64_t>(actual[i])==std::bit_cast<uint64_t>(reference[i]),"small covariance ordered/edge parity");
    }
  }
#endif
  wide_grid_threshold_parity();
  {
    constexpr size_t rows=131,columns=67;
    std::mt19937 rng(839); std::normal_distribution<float> normal;
    std::vector<float> values(rows*columns),mean,sigma,expected_mean,expected_sigma;
    for (auto& value:values) value=normal(rng);
    tieredq_detail::projection_stats_reference(values,rows,columns,expected_mean,expected_sigma);
    tieredq_detail::projection_stats(values,rows,columns,mean,sigma);
    for (size_t i=0;i<columns;++i) {
      require(std::bit_cast<uint32_t>(mean[i])==std::bit_cast<uint32_t>(expected_mean[i]),"projected mean order");
      require(std::bit_cast<uint32_t>(sigma[i])==std::bit_cast<uint32_t>(expected_sigma[i]),"projected variance order");
    }
  }
  for (unsigned bits=1;bits<=8;++bits) {
    const auto& levels=tieredq_detail::gaussian_codebooks()[bits];
    require(levels.size()==size_t(1u<<bits),"canonical codebook length");
    require(std::is_sorted(levels.begin(),levels.end()),"canonical codebook order");
    require(tieredq_detail::distortion(levels)>0,"canonical codebook distortion");
  }
  training_covariance_layout();
  centroid_update_parity();
  using namespace tieredq_detail;
  for (unsigned bits=1;bits<=8;++bits) {
    std::vector<double> thresholds((1u<<bits)-1);
    // Repeated thresholds also occur in valid, non-strictly sorted persisted codebooks.
    for (size_t i=0;i<thresholds.size();++i) thresholds[i]=double(i/2)-64;
    for (double threshold:thresholds) for (double value:{std::nextafter(threshold,-INFINITY),threshold,std::nextafter(threshold,INFINITY)}) {
      unsigned expected=0; while (expected<thresholds.size() && value>thresholds[expected]) ++expected;
      require(quantize_code(value,thresholds)==expected,"exact threshold boundary");
    }
  }
  const std::vector<float> center{0},basis{1},mean{0},sigma{1};
  std::vector<std::vector<float>> levels(9); levels[2]={-1.5f,-.5f,.5f,1.5f};
  const std::vector<TieredQTier> tiers{{0,1,2,0,0}};
  const EncodingModel model{1,center,mean,sigma,levels,tiers};
  EncodingWorkspace work(model); work.v[0]=1; work.z[0]=1;
  require(scale_grid_search(model,1,work)==.5,"first grid point dot on equal energies");
  require(work.chosen_codes[0]==2,"first grid point codes on equal energies");
}
void encoder_reductions() {
  using namespace tieredq_detail;
  constexpr size_t d=67,m=65,rows=7;
  std::mt19937 rng(314); std::normal_distribution<float> normal;
  std::vector<float> basis(d*m);
  std::vector<double> input(rows*d);
  for (auto& value:basis) value=normal(rng);
  for (auto& value:input) value=normal(rng);
  for (bool transpose:{false,true}) {
    const auto panels=encoder_panels(basis,d,m,transpose);
    const size_t depth=transpose ? m : d,columns=transpose ? d : m,stride=panels.padded_columns();
    std::vector<double> actual(rows*stride);
    for (bool centroid:{false,true}) for (auto kernel:{TieredQKernel::scalar,TieredQKernel::avx2,TieredQKernel::avx512}) {
      if (!TieredQ::kernel_available(kernel)) continue;
      if (centroid) ordered_matrix<true>(input.data(),rows,panels,actual.data(),kernel);
      else ordered_matrix(input.data(),rows,panels,actual.data(),kernel);
      for (size_t i=0;i<rows;++i) for (size_t j=0;j<columns;++j) {
        double sum=0;
        for (size_t k=0;k<depth;++k) {
          const double b=transpose ? basis[j*m+k] : basis[k*m+j];
          sum+=centroid ? b*b/2-input[i*depth+k]*b : input[i*depth+k]*b;
        }
        require(std::bit_cast<uint64_t>(actual[i*stride+j])==std::bit_cast<uint64_t>(sum),"matrix reduction order");
      }
    }
    for (auto kernel:{TieredQKernel::avx2,TieredQKernel::avx512})
      if (!TieredQ::kernel_available(kernel)) rejects([&]{ordered_matrix(input.data(),rows,panels,actual.data(),kernel);});
  }
}
void grid_kernels() {
  using namespace tieredq_detail;
  constexpr size_t d=67,m=65,rows=7,centroids=3;
  std::mt19937 rng(123); std::normal_distribution<float> normal;
  std::vector<float> basis(d*m),centers(centroids*d),mean(m),sigma(m),input(rows*d);
  for (auto* values:{&basis,&centers,&mean,&input}) for (auto& value:*values) value=normal(rng);
  for (size_t r=0;r<m;++r) sigma[r]=float(r%7+1);
  std::vector<std::vector<float>> levels(9);
  std::vector<TieredQTier> tiers;
  uint32_t rank=0,offset=0;
  for (unsigned bits=1;bits<=8;++bits) {
    for (unsigned code=0;code<(1u<<bits);++code) levels[bits].push_back(float(code)*.1f-float(1u<<bits)*.05f);
    const unsigned dimensions=bits==1 ? 9 : 8;
    tiers.push_back({rank,dimensions,bits,0,offset}); rank+=dimensions; offset+=dimensions*bits;
  }
  const EncodingModel model{d,centers,mean,sigma,levels,tiers};
  EncodingWorkspace work(model);
  for (const auto& tier:tiers) for (size_t j=0;j<tier.dimensions;++j) {
    const size_t r=tier.start+j; const auto& thresholds=work.thresholds[tier.bits];
    const double threshold=thresholds[j%thresholds.size()];
    work.z[r]=j%3==0 ? threshold : std::nextafter(threshold,j%3==1 ? -INFINITY : INFINITY);
    work.v[r]=normal(rng);
  }
  std::array<double,scale_grid_storage> reference_dots{},reference_norms{},dots{},norms{};
  scale_grid_terms(model,work,reference_dots.data(),reference_norms.data(),TieredQKernel::scalar);
  for (auto kernel:{TieredQKernel::scalar,TieredQKernel::avx2,TieredQKernel::avx512}) {
    if (!TieredQ::kernel_available(kernel)) continue;
    scale_grid_terms(model,work,dots.data(),norms.data(),kernel);
    for (size_t point=0;point<scale_grid_points;++point) {
      require(std::bit_cast<uint64_t>(dots[point])==std::bit_cast<uint64_t>(reference_dots[point]),"grid dot order/boundary");
      require(std::bit_cast<uint64_t>(norms[point])==std::bit_cast<uint64_t>(reference_norms[point]),"grid norm order/boundary");
    }
  }
}
void half_tests() {
  using namespace tieredq_detail;
  for (unsigned bits=0;bits<=UINT16_MAX;++bits) {
    if ((bits&0x7c00)==0x7c00) continue;
    require(half_encode(half_decode(uint16_t(bits)))==bits,"binary16 round trip");
  }
  for (unsigned bits=0;bits<0x7bff;++bits) {
    const double midpoint=(double(half_decode(uint16_t(bits)))+half_decode(uint16_t(bits+1)))/2;
    require(half_encode(midpoint)==bits+(bits&1),"binary16 midpoint RNE");
    require(half_encode(std::nextafter(midpoint,0.))==bits,"binary16 below midpoint");
    require(half_encode(std::nextafter(midpoint,INFINITY))==bits+1,"binary16 above midpoint");
  }
  require(half_encode(1+std::ldexp(1.,-11))==0x3c00,"half tie even down");
  require(half_encode(1+3*std::ldexp(1.,-11))==0x3c02,"half tie even up");
  require(half_encode(std::ldexp(1.,-25))==0,"half subnormal tie");
  require(half_encode(65504)==0x7bff,"half max");
  require(half_encode(-0.)==0x8000,"half signed zero");
  rejects([]{half_encode(65505);}); rejects([]{half_encode(INFINITY);}); rejects([]{half_encode(NAN);});
}
void parity(const std::string& prefix,unsigned d,unsigned bits) {
  constexpr size_t rows=81;
  auto body=model(d,bits); persist(prefix,body.data); auto codec=TieredQ::load(prefix);
  require(codec.metadata_bytes_per_vector()==5,"metadata size");
  codec.save(prefix+".roundtrip"); require(read_bytes(prefix)==read_bytes(prefix+".roundtrip"),"model round trip");
  std::mt19937 rng(42); std::normal_distribution<float> normal;
  std::vector<float> x(rows*d),q(d),decoded(rows*d),external(rows);
  for (auto& value:x) value=normal(rng); for (auto& value:q) value=normal(rng);
  auto codes=codec.encode(x.data(),rows); codec.decode(codes.data(),rows,decoded.data());
  auto again=codes; codec.encode(x.data(),rows,again.data(),external.data()); require(again==codes,"correction output changes codes");
  tieredq_detail::Layout layout{codec.code_bytes()};
  std::vector<uint8_t> planes(((rows+15)/16)*16*codec.bytes_per_vector()),unpacked(codes.size());
  codec.pack(codes.data(),rows,planes.data()); codec.unpack(planes.data(),rows,unpacked.data()); require(codes==unpacked,"pack/unpack");
  for (size_t i=0;i<rows;++i) {
    const auto* doc=codes.data()+i*codec.bytes_per_vector();
    require(layout.half(doc,layout.correction_offset())==tieredq_detail::half_encode(2*(double(x[i*d])-decoded[i*d])),"rounded correction reference");
    require(external[i]==layout.correction(doc),"correction output");
  }
  std::vector<float> reference(rows),scores(rows),scalar(rows),approx(rows);
  for (size_t i=0;i<rows;++i) {
    double dot=0; for (size_t j=0;j<d;++j) dot+=double(q[j])*decoded[i*d+j];
    reference[i]=float(dot)+q[0]/2*external[i];
  }
  auto query=codec.query(q.data());
  codec.score(codes.data(),rows,query,scalar.data(),TieredQKernel::scalar);
  for (auto kernel:{TieredQKernel::scalar,TieredQKernel::avx2,TieredQKernel::avx512}) {
    if (!TieredQ::kernel_available(kernel)) continue;
    codec.score(codes.data(),rows,query,scores.data(),kernel);
    for (size_t i=0;i<rows;++i) require(std::fabs(scores[i]-scalar[i])<.002f,"row kernel parity");
    codec.score_planes(planes.data(),rows,query,scores.data(),kernel);
    for (size_t i=0;i<rows;++i) require(std::fabs(scores[i]-scalar[i])<.002f,"plane kernel parity");
  }
  for (size_t i=0;i<rows;++i) require(std::fabs(scalar[i]-reference[i])<.003f,"independent score reference");
  TieredQIndex index(codec,dv::TieredQIndexMode::full); index.add(x.data(),rows); index.save(prefix+".index");
  auto loaded=TieredQIndex::load(prefix+".index");
  require(loaded.search(q.data(),rows).size()==rows,"index load"); loaded.save(prefix+".index2");
  require(read_bytes(prefix+".index")==read_bytes(prefix+".index2"),"index bytes");
  auto corrupt=codes; corrupt[codec.code_bytes()]=1;
  persist(prefix+".bad",body.data,corrupt,rows,true); rejects([&]{TieredQIndex::load(prefix+".bad");});
  corrupt=codes; corrupt[layout.factor_offset()]=0; corrupt[layout.factor_offset()+1]=0x7c;
  persist(prefix+".bad",body.data,corrupt,rows,true); rejects([&]{TieredQIndex::load(prefix+".bad");});
  corrupt=codes; corrupt[layout.correction_offset()]=0; corrupt[layout.correction_offset()+1]=0x7e;
  persist(prefix+".bad",body.data,corrupt,rows,true); rejects([&]{TieredQIndex::load(prefix+".bad");});
  if ((d*bits)%8) { corrupt=codes; corrupt[codec.code_bytes()-1]|=0x80; persist(prefix+".bad",body.data,corrupt,rows,true); rejects([&]{TieredQIndex::load(prefix+".bad");}); }
  for (size_t offset:{size_t(4*4),size_t(6*4),size_t(32+12),size_t(32+16)}) {
    auto invalid=body.data; invalid[offset]^=1;
    persist(prefix+".bad",invalid); rejects([&]{TieredQ::load(prefix+".bad");});
  }
  auto invalid=body.data;
  std::fill_n(invalid.begin()+32+20,8,0xff);
  persist(prefix+".bad",invalid); rejects([&]{TieredQ::load(prefix+".bad");});
  corrupt=codes; corrupt.pop_back(); persist(prefix+".bad",body.data,corrupt,rows,true);
  rejects([&]{TieredQIndex::load(prefix+".bad");});
  std::vector<float> center(d,0); center[0]=2;
  const auto zero=codec.encode(center.data(),1);
  require(layout.factor(zero.data())==0 && layout.correction(zero.data())==0,"zero residual metadata");
  for (size_t j=0;j<d;++j) {
    const size_t bit=j*bits;
    unsigned code=zero[bit/8]>>(bit%8);
    if (bit%8+bits>8) code|=unsigned(zero[bit/8+1])<<(8-bit%8);
    require((code&((1u<<bits)-1))==(1u<<(bits-1))-1,"zero threshold chooses lower level");
  }
  std::vector<float> huge(rows*d,1e10f); rejects([&]{codec.encode(huge.data(),rows);});
}
// Loaded models omit encoder state until encode(); the lazily built state encodes identically.
void lazy_encoder() {
  constexpr size_t d=24,rows=64;
  std::mt19937 rng(5); std::normal_distribution<float> normal;
  std::vector<float> sample(rows*d);
  for (auto& value:sample) value=normal(rng);
  const auto trained=TieredQ::train(TieredQConfig{.dimensions=uint32_t(d),.code_bits=uint32_t(d*8),.num_centroids=8,.rotate_tiers=true},sample.data(),rows,42);
  const size_t trained_serving=trained.memory_bytes();
  const auto expected=trained.encode(sample.data(),rows);
  const size_t encoder=trained.memory_bytes()-trained_serving;
  require(encoder>0,"encode() builds encoder state");
  auto loaded=TieredQ::deserialize(trained.serialize());
  const auto copy=loaded;
  const size_t loaded_serving=loaded.memory_bytes(),copy_serving=copy.memory_bytes();
  constexpr size_t concurrent_encodes=4;
  std::vector<std::future<std::vector<uint8_t>>> encodes;
  for (size_t worker=0;worker<concurrent_encodes;++worker)
    encodes.push_back(std::async(std::launch::async,[&,model=loaded] { return model.encode(sample.data(),rows); }));
  for (auto& encoded:encodes) require(encoded.get()==expected,"concurrent lazy encoder parity");
  require(loaded.memory_bytes()-loaded_serving==encoder,"loaded models build encoder state on encode()");
  require(copy.memory_bytes()-copy_serving==encoder,"copies share the built encoder");
}
// Multi-tier model: the grouped FP32 plane scan matches the scalar scorer, and its plan is validated.
void grouped_scan() {
  constexpr size_t d=64,rows=150,sample_rows=512;
  std::mt19937 rng(11); std::normal_distribution<float> normal;
  std::vector<float> sample(sample_rows*d),q(d);
  for (size_t i=0;i<sample_rows;++i) for (size_t j=0;j<d;++j) sample[i*d+j]=normal(rng)*(1+float(d-j)/8);
  for (auto& value:q) value=normal(rng);
  auto codec=TieredQ::train(TieredQConfig{.dimensions=uint32_t(d),.code_bits=uint32_t(d*4),.num_centroids=16,.rotate_tiers=true},sample.data(),sample_rows,42);
  require(codec.tiers().size()>2,"grouped scan needs several tiers");
  auto codes=codec.encode(sample.data(),rows);
  std::vector<uint8_t> planes(((rows+15)/16)*16*codec.bytes_per_vector()); codec.pack(codes.data(),rows,planes.data());
  auto query=codec.query(q.data());
  require(!query.scan_groups.empty(),"FP32 queries carry scan groups");
  bool straddles=false;
  for (const auto& g:query.scan_groups) straddles|=g.bit_offset%32+g.bits>32;
  require(straddles,"grouped scan exercises window-straddling groups");
  std::vector<tieredq_detail::ScanGroupPlan> plan;
  tieredq_detail::build_scan_plan(codec.tiers(),plan);
  require(plan.size()==query.scan_groups.size(),"scan plan covers every group");
  for (size_t i=0;i<plan.size();++i) {
    const auto& g=query.scan_groups[i]; const auto& p=plan[i];
    require(p.window==g.bit_offset/32 && p.shift==g.bit_offset%32 && p.left==32-p.shift &&
            p.first==g.first && p.bits==g.bits && p.table==g.table && p.straddles==(p.shift+p.bits>32),
            "scan plan geometry matches groups");
  }
  std::vector<float> scalar(rows),scores(rows);
  codec.score(codes.data(),rows,query,scalar.data(),TieredQKernel::scalar);
  for (auto kernel:{TieredQKernel::avx2,TieredQKernel::avx512}) {
    if (!TieredQ::kernel_available(kernel)) continue;
    codec.score_planes(planes.data(),rows,query,scores.data(),kernel);
    for (size_t i=0;i<rows;++i) require(std::fabs(scores[i]-scalar[i])<=1e-4f*(1+std::fabs(scalar[i])),"grouped plane scan parity");
  }
  // search() filters chunks against the heap minimum; it must equal a full sort of the same kernel's scores.
  codec.score_planes(planes.data(),rows,query,scores.data());
  std::vector<uint32_t> order(rows); std::iota(order.begin(),order.end(),0u);
  std::stable_sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b){ return scores[a]>scores[b]; });
  TieredQIndex index(codec,dv::TieredQIndexMode::full); index.add(sample.data(),rows);
  constexpr size_t k=10; const auto hits=index.search(q.data(),k);
  for (size_t i=0;i<k;++i) require(hits[i].id==order[i] && hits[i].score==scores[order[i]],"filtered search equals full sort");
  auto tampered=query; tampered.scan_tables[0]+=1;
  rejects([&]{codec.score_planes(planes.data(),rows,tampered,scores.data());});
  tampered=query; tampered.scan_groups.pop_back();
  rejects([&]{codec.score_planes(planes.data(),rows,tampered,scores.data());});
}
}
void wide_dims_cap() {
  constexpr size_t d=16,rows=128;
  std::vector<float> sample(rows*d); std::mt19937 rng(9); std::normal_distribution<float> normal;
  for (auto& value:sample) value=normal(rng);
  auto capped=TieredQ::train(TieredQConfig{.dimensions=d,.code_bits=d*6+4,.num_centroids=4,.rotate_tiers=true,.max_bits=7,.wide_dims=4},sample.data(),rows,42);
  bool boosted=false;
  for (const auto& tier:capped.tiers()) if (tier.bits>6) {
    boosted=true; require(tier.start+tier.dimensions<=4,"wide_dims must cap high-bit tiers to the top ranks");
  }
  require(boosted,"wide_dims=4 with max_bits=7 must boost some coordinate");
  auto capped6=TieredQ::train(TieredQConfig{.dimensions=d,.code_bits=d*6,.num_centroids=4,.rotate_tiers=true,.max_bits=7,.wide_dims=0},sample.data(),rows,42);
  for (const auto& tier:capped6.tiers()) require(tier.bits<=6,"wide_dims=0 must cap at 6 bits");
}
// Format-3 models store a binary16 basis. Rewriting it as a format-2 FP32 basis holding the same rounded
// values must leave encoding, exact search and fast search unchanged, while each format round-trips.
void binary16_basis_equivalence() {
  constexpr size_t d=40,rows=300,documents=513,k=10;
  std::mt19937 rng(11); std::normal_distribution<float> normal;
  std::vector<float> sample(rows*d),docs(documents*d),query(d);
  for (auto* values:{&sample,&docs,&query}) for (auto& value:*values) value=normal(rng);
  for (unsigned bits:{1u,2u}) {
    const auto half_model=TieredQ::train(TieredQConfig{.dimensions=uint32_t(d),.code_bits=uint32_t(d*bits),.num_centroids=8,.rotate_tiers=true},sample.data(),rows,3);
    const auto bytes=half_model.serialize();
    auto u32=[&](size_t at) { uint32_t value=0; for (size_t i=0;i<4;++i) value|=uint32_t(bytes[at+i])<<(8*i); return value; };
    auto u64=[&](size_t at) { return uint64_t(u32(at))|uint64_t(u32(at+4))<<32; };
    constexpr size_t format_offset=16,tier_count_offset=28;
    require(u32(format_offset)==TieredQ::model_format,"trained models use the binary16 format");
    const size_t centroids_offset=tier_count_offset+4+u32(tier_count_offset)*5*4;
    const size_t basis_offset=centroids_offset+8+u64(centroids_offset)*4,basis_count=u64(basis_offset);
    Bytes fp32; fp32.data.assign(bytes.begin(),bytes.begin()+basis_offset);
    for (size_t i=0;i<4;++i) fp32.data[format_offset+i]=uint8_t(TieredQ::fp32_basis_model_format>>(8*i));
    std::vector<float> rounded(basis_count);
    for (size_t i=0;i<basis_count;++i)
      rounded[i]=tieredq_detail::half_decode(uint16_t(bytes[basis_offset+8+2*i]|bytes[basis_offset+9+2*i]<<8));
    fp32.floats(rounded);
    fp32.data.insert(fp32.data.end(),bytes.begin()+basis_offset+8+2*basis_count,bytes.end());
    const auto fp32_model=TieredQ::deserialize(fp32.data);
    require(fp32_model.serialize()==fp32.data,"format-2 round trip");
    require(TieredQ::deserialize(bytes).serialize()==bytes,"format-3 round trip");
    // Rank panels pad the binary16 basis to whole 16-rank panels.
    const size_t panel_halves=tieredq_detail::basis_panel_size(fp32_model.model_dimensions(),fp32_model.coded_dimensions());
    require(TieredQ::deserialize(bytes).memory_bytes()+basis_count*4==fp32_model.memory_bytes()+panel_halves*2,"binary16 basis halves its memory");
    require(half_model.encode(docs.data(),documents)==fp32_model.encode(docs.data(),documents),"binary16 basis changes encoding");
    for (auto mode:{TieredQIndexMode::fast,TieredQIndexMode::full}) {
      TieredQIndex half_index(half_model,mode),fp32_index(fp32_model,mode);
      half_index.add(docs.data(),documents); fp32_index.add(docs.data(),documents);
      for (const auto& [half_hits,fp32_hits]:{std::pair{half_index.search(query.data(),k),fp32_index.search(query.data(),k)},
                                              std::pair{half_index.search_fast(query.data(),k),fp32_index.search_fast(query.data(),k)}})
        for (size_t i=0;i<k;++i)
          require(half_hits.at(i).id==fp32_hits.at(i).id && half_hits[i].score==fp32_hits[i].score,"binary16 basis changes search");
    }
  }
}
void tieredq_encoder_tests();
void tieredq_model_tests(const std::string& prefix) {
  half_tests(); encoder_boundaries(); encoder_reductions(); grid_kernels(); tieredq_encoder_tests();
  parity(prefix+".model",7,1); parity(prefix+".model",16,4); parity(prefix+".model",65,2);
  for (unsigned bits:{3u,5u,6u,7u,8u}) parity(prefix+".model",16,bits);
  grouped_scan(); wide_dims_cap(); lazy_encoder(); binary16_basis_equivalence();
  parity(prefix+".model",3072,2); parity(prefix+".model",3072,4);
  std::vector<float> sample(128*16); std::mt19937 rng(7); std::normal_distribution<float> normal;
  for (auto& value:sample) value=normal(rng);
  auto trained=TieredQ::train(TieredQConfig{.dimensions=16,.code_bits=37,.num_centroids=4,.rotate_tiers=true},sample.data(),128,42);
  trained.save(prefix+".trained"); auto loaded=TieredQ::load(prefix+".trained");
  require(trained.model_id()==loaded.model_id(),"trained roundtrip");
  require(loaded.config().rotate_tiers,"training rotation flag");
  const size_t retained=trained.coded_dimensions();
  const size_t expected_bytes=8*4+trained.tiers().size()*5*4+4*8+4+9*8+4*(4*16+2*retained+510)+2*16*retained;
  require(trained.model_bytes()==expected_bytes,"compact sections");
  const auto encoded=trained.encode(sample.data(),128);
  require(encoded==loaded.encode(sample.data(),128),"loaded encoder");
  std::vector<float> decoded(sample.size()); loaded.decode(encoded.data(),128,decoded.data());
  const auto bytes=read_bytes(prefix+".trained");
  const size_t centers_offset=80+8*4+trained.tiers().size()*5*4+8;
  const tieredq_detail::Layout layout{trained.code_bytes()};
  for (size_t row=0;row<128;++row) {
    const auto* doc=encoded.data()+row*trained.bytes_per_vector();
    double correction=0;
    for (size_t j=0;j<16;++j) {
      const size_t offset=centers_offset+(size_t(doc[trained.code_bytes()])*16+j)*4;
      uint32_t value=0; for (size_t byte=0;byte<4;++byte) value|=uint32_t(bytes[offset+byte])<<(byte*8);
      correction+=(double(sample[row*16+j])-decoded[row*16+j])*std::bit_cast<float>(value);
    }
    require(layout.half(doc,layout.correction_offset())==tieredq_detail::half_encode(correction),"rotated correction uses decoded reconstruction");
  }
}
