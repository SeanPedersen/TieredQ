// Independent direct caller of the official IVF API for adapter parity checks.
#include <rabitqlib/index/ivf/ivf.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__OPTIMIZE_SIZE__)
#error "Reference kernels require -O3"
#endif
namespace {
constexpr size_t top_k = 10;
template<class T> std::vector<T> read(const std::string& path, size_t count) {
  std::vector<T> values(count);
  std::ifstream file(path, std::ios::binary);
  if (!file.read(reinterpret_cast<char*>(values.data()), count*sizeof(T)))
    throw std::runtime_error("Cannot read " + path);
  return values;
}
}
int main(int argc, char** argv) {
  try {
    constexpr int argument_count = 10;
    if (argc != argument_count) throw std::runtime_error("usage: reference BASE QUERY OUT N D NQ BITS l2|ip THREADS");
    const size_t n=std::stoul(argv[4]), d=std::stoul(argv[5]), nq=std::stoul(argv[6]), bits=std::stoul(argv[7]);
    omp_set_num_threads(std::stoi(argv[9]));
    const auto base=read<float>(argv[1],n*d), queries=read<float>(argv[2],nq*d);
    const auto metric=std::string(argv[8])=="l2" ? rabitqlib::METRIC_L2 : rabitqlib::METRIC_IP;
    std::vector<float> centroid(d,0);
    for (size_t i=0;i<n;++i) for (size_t j=0;j<d;++j) centroid[j]+=base[i*d+j]/n;
    std::vector<uint32_t> assignments(n,0), ids(nq*top_k);
    rabitqlib::ivf::IVF index(n,d,1,bits,metric);
    index.construct(base.data(),centroid.data(),assignments.data(),false);
    for (size_t i=0;i<nq;++i) index.search(queries.data()+i*d,top_k,1,ids.data()+i*top_k,true);
    std::ofstream output(argv[3],std::ios::binary);
    if (!output.write(reinterpret_cast<const char*>(ids.data()),ids.size()*sizeof(uint32_t)))
      throw std::runtime_error("Cannot write results");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
