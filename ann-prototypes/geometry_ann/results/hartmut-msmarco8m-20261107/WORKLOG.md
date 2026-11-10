# Active task handoff / progress

User authorized: finish all requested configs on full MSMARCO Qwen normalized
8,840,823 x 1024 HDF5, now using INT8 scalar quantization; copy INT8 to NVMe;
report recall, query latency, build time, serving RAM and build RSS. Do not
substitute a 1M subset. Overall goal: fast build / low RAM / high-recall fast
search, beating IVFFlat. No win has yet been established. Do not claim results
for the large corpus before jobs finish.

Remote via ssh sean@hartmut. Remote AGENTS: prod in services/, experiments/tests
and scripts under ~/experiments (keep home clean). Big data goes to /mnt/storage.
Main root: /home/sean/experiments/TieredQ-issue44-msmarco8m-20261107.
Big root: /mnt/storage/tieredq-issue44-msmarco8m-20261107.
Main nohup job PID is in main-root/job.pid; status.json gives phase/model.
job.exitcode is written on exit. Current phase: waiting_for_download.
Source download was ~25.4 GiB / 33.72 GiB at 2026-11-07 23:26 UTC.
Native hf process uses authenticated existing HF credentials, but all large
caches/download files are on storage. DO NOT print token files or Xet logs,
which may contain signed URLs. No need to restart the working download.
Sequential incomplete file is under big-root/xet-download/.cache/huggingface/download/.
Final file will be big-root/xet-download/msmarco-qwen-1024-normalized.hdf5.
A fallback safely resumes its sequential prefix with aria2 if the native client
exits early. All source bytes must match SHA256
 efa21f8418e29137ad5be44171fd4657e1738caee86eb62645ad20605457cb2c.

job.sh waits for complete source, runs SHA-verified prepare_hdf5_view.py --int8
with 250 calibration / 750 held-out queries, copies 9.053 GB codes and FP32
scale/offset tables to NVMe experiment root with hashes and 2 GiB free headroom,
then runs study.py. Old 4.133 GB 1M NPZ cache was moved to
/mnt/storage/tieredq-dataset-cache with SHA256 verification and a symlink at the
original path; cache-relocation.json records it. NVMe free ~12.9 GB before
SIFT SQ8 control (128 MB); enough for codes and headroom.

study.py pins CPU 2 and serially runs: FP32 exact ground-truth diagnostic on
50 calibration queries; SQ8 exhaustive diagnostic on same queries; adaptive4,
fixed4 (graph + flat), fixed12 (graph + flat); Faiss shared-codec IVF-SQ8;
Faiss original FP32 IVFFlat. 8192 leaves/centroids, 64 coarse parents / 128 fixed
children. Calibration caps: 35K/50K plus 1/2/3.5/5% of full base; graph metadata
512/1024/2048/4096/8256; flat all. Both frozen targets 0.95/0.99/0.995 and
original SIFT operating points are replayed, three query repeats on ONE build.
Failures recorded, no retuning on test queries. Query memory/swap/major faults
are recorded; flag paging if seen. Faiss serialization counts via bounded
callback instead of copying the whole index.

INT8 codec is per-coordinate affine full-population min/max. Native prototype
scans signed codes with AVX-512/FMA filter and exact double rescoring of
competitive DECODED-code distances. Queries/centroids stay FP32. Original FP32
HDF5 is mapped for partition training; it is not kept for query rescoring.
Inputs unmap after build. Therefore serving vector RAM is 4x smaller, but
training RSS is NOT promised 4x smaller. Geometry bounds disabled (not inflated
for quantization). Exact AVX2 four-double-lane centroid kernel now also trains
partitions/graph; verified bit-equivalent to old metric. Packing streams source
row order to avoid HDD random I/O. Production TieredQ untouched.

Small C++ tests, mapped-input tests, SQ8 exact reconstruction/tails/scale/offset
and cap checks, and full toy preparation/selection/Faiss replay pass on hartmut.
Local ASan/UBSan pass. A local preflight accidentally recreated ignored .venv;
it was rebuilt from project configuration and subsequently isolated --no-project
runs are used. This was acknowledged to user.

SIFT SQ8 same-implementation control is COMPLETE at
/home/sean/experiments/TieredQ-issue44-sift-sq8-20261107; data/source/build linked
to main study. Its results were copied to sibling repository artifact directory
hartmut-sift-sq8-20261107 with README/summary. FP32 exhaustive 50q recall 1.000;
SQ8 exhaustive 0.986. Fixed4 original 35K/2048 graph point: recall .9815,
p50 1.656 ms, build 15.75 s, estimated owned 138.27 MB, serving RSS 173.98 MB,
build RSS 819.50 MB. Faiss SQ8 nprobe96: .9589 / .515 ms, serving RSS 299.96 MB;
nprobe256 diagnostic: .9828 / 1.025 ms. 0.99/0.995 targets unreachable in SQ8 grid;
fixed4 0.95 calibration-selected point misses held-out target (.9438). SIFT
integer data could use a lossless byte grid; this control uses generic min/max
SQ8 for Qwen comparability. No claim of inherent INT8 accuracy ceiling.

Earlier FP32 SIFT results posted explicitly at user's request:
https://github.com/SeanPedersen/TieredQ/issues/44#issuecomment-6713726764
Do not post additional issue messages without authorization. Earlier artifact
folders hartmut-routing-20261107, hartmut-simd-20261107, simd-scan-20261107.
Git working tree has all prototype changes and uncommitted results; started
clean at commit 0128676. Do not discard any working changes.

Next: monitor main download/status/job.pid (no need to restart), finish all
large-corpus runs, inspect stdout CSVs, memory CSVs, exit codes, use summarize.py
on remote root or locally copied artifacts. Copy only small results/logs/manifest,
not large vectors. Add final measured scaling table + RAM/precision caveats to
README, report to user. Do not claim experiments complete on mere launch.

## Native INT8 training follow-up

Full download SHA256 passed; conversion and verified NVMe copy completed.
Payload 9,053,002,752 bytes; conversion 938.25 s and peak RSS 209,063,936 bytes.
Earlier descriptions of FP32-trained compressed models are superseded: both
prototype and Faiss compressed configurations now train on decoded INT8 values.
C++ decodes into registers; Faiss decodes bounded samples/batches.
The initial exhaustive diagnostic was stopped before any index build to replace
its repeated disk passes with one batched corpus pass. Its partial output is
archived separately. SIFT controls are refreshed before the large study.

## Completion

All full-corpus stages returned zero; finished 2026-11-08 02:55:48 UTC.
Raw small artifacts copied locally and summary.py validation passed: all query
repeat groups complete and hard caps respected. README records measured table,
recall misses, index size versus process RSS distinction, and FP32 swap caveat.
No full-scale conclusion is based on SIFT results.
