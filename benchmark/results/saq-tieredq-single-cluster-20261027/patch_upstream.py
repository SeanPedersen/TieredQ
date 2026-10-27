"""Add allocation observations and correct a last-full-block scan bug; preserve SAQ kernels."""
from pathlib import Path
import sys
root=Path(sys.argv[1])
p=root/'saqlib/quantization/cluster_data.hpp'
s=p.read_text()
needle='    ~SaqCluData() {'
assert s.count(needle)==1
method='''    size_t benchmark_owned_bytes() const {
        auto aligned = [](size_t n) { return utils::rd_up_to_multiple_of(n, size_t(64)); };
        size_t total = aligned(shortb_factors_fcnt_ * num_blocks_ * sizeof(float));
        total += aligned(shortb_code_bytes_ * num_blocks_);
        total += aligned(longb_code_bytes_tot_);
        total += aligned(num_vec_ * num_segments_ * sizeof(ExFactor));
        total += aligned(ids_.capacity() * sizeof(PID));
        total += segments_.capacity() * sizeof(CAQClusterData);
        for (const auto &segment : segments_) total += segment.centroid().size() * sizeof(float);
        return total;
    }

'''
p.write_text(s.replace(needle,method+needle))
p=root/'saqlib/quantization/saq_searcher.hpp';s=p.read_text()
old='auto curr_num_points = (blk_idx == num_blocks - 1) ? clusters->num_vec_ % KFastScanSize : KFastScanSize;'
assert s.count(old)==1
s=s.replace(old,'auto curr_num_points = std::min(KFastScanSize, clusters->num_vec_ - blk_idx * KFastScanSize);')
p.write_text(s)

p=root/'saqlib/index/ivf.hpp';s=p.read_text()
s=s.replace('    QuantMetrics quant_metrics_;', '    double benchmark_encode_seconds = 0;\n    QuantMetrics quant_metrics_;',1)
s=s.replace('        auto tm_ms = stopw.getElapsedTimeMicro() / 1000.0;', '        auto tm_ms = stopw.getElapsedTimeMicro() / 1000.0;\n        benchmark_encode_seconds = tm_ms / 1000.0;',1)
p.write_text(s)

p=root/'saqlib/quantization/caq/caq_encoder.hpp';s=p.read_text()
old='''            base_code.norm_ip_o_oa = caq.ip_o_oa / caq.o_l2norm / std::sqrt(caq.oa_l2sqr);

            base_code.code = std::move(caq.code);
        }'''
new='''            base_code.norm_ip_o_oa = caq.ip_o_oa / caq.o_l2norm / std::sqrt(caq.oa_l2sqr);
        }
        if (num_bits_ > 0) {
            base_code.code = std::move(caq.code);
        }'''
assert s.count(old)==1
p.write_text(s.replace(old,new))
