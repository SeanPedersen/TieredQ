# Frieda: current TieredQ IVF versus zvec IVF_RABITQ

Measured on 2026-11-06 (Europe/Berlin). TieredQ was built from upstream `main`
at `748a79495b7e498364ea99ad1caaa5269cd46bbf`, verified against GitHub before
the run. The installed distribution version is 0.1.1; the commit identifies the
actual implementation. zvec is 0.8.0, matching the earlier Hartmut comparison.

## Result

**TieredQ does not yet beat zvec across recall targets on Frieda.** Among the
tested configurations, it is 1.09× faster at recall@10 ≥ 0.80, but zvec is
1.44–3.81× faster at the tested 0.90–0.99 targets, including nested IVF.
The smallest qualifying TieredQ indexes use 25–46% fewer compressed index
bytes than the smallest qualifying zvec indexes across all six recall targets.
These byte figures are index accounting, not process RAM measurements.

| Recall@10 target | Fastest zvec QPS | Fastest TieredQ QPS | TieredQ/zvec speed |
|---|---:|---:|---:|
| 0.80 | 4,832 | 5,276 | 1.09× |
| 0.90 | 3,985 | 2,776 | 0.70× |
| 0.95 | 3,312 | 870 | 0.26× |
| 0.97 | 2,510 | 715 | 0.28× |
| 0.98 | 2,510 | 715 | 0.28× |
| 0.99 | 1,023 | 615 | 0.60× |

The 80% result is a modest speed advantage: its three TieredQ passes delivered
5,270–5,281 QPS versus zvec's 4,659–4,887 QPS. It uses 132.6 bytes/vector versus
340.2 for the fastest qualifying zvec configuration. At higher recall, nested
IVF improves over the wide flat codes but does not catch zvec.

See [comparison.md](comparison.md) for winning configurations, index sizes,
and comparisons constrained by index size; [curves.md](curves.md) contains
every measured recall/QPS curve. The complete sweep finished with exit code 0:
29 index builds, 203 search configurations, and 609 timed passes, taking
42 minutes 28 seconds. All TSV medians, index byte counts, and recalls were
checked against the per-pass JSONL records; see
[result-validation.txt](result-validation.txt). Across cells, the median ratio
of fastest to slowest pass was 1.023; the worst was 1.278. The large higher-recall
speed gaps exceed the observed timing variation, but this remains one build
per configuration on one shared VM.

## Method

- Frieda (`vps-9ad39dea`): AMD EPYC Milan KVM VM, AVX2, four vCPUs, 7.6 GiB RAM;
  GCC 15.2.0, Python 3.12.16, NumPy 2.5.4. See [environment.txt](environment.txt).
- DBpedia Gemini-768: 99,000 base rows, 1,000 held-out document queries,
  inner-product top-10 ground truth. All three input files have the same SHA-256
  checksums as the previous Hartmut IVF comparison.
- Each engine trains its own 256 coarse lists, using its shipped Python API.
  `nprobe` is 4, 8, 16, 32, 64, 128, or 256.
- One query per Python call, pinned to vCPU 3; OpenMP, OpenBLAS, and MKL thread
  counts set to one. Configurations run sequentially in separate processes.
- Each configuration has three timing passes over all 1,000 queries, each with
  100 warmup queries excluded from timing. TSV QPS is the median; JSONL files
  preserve every pass and its recall. Recall must agree across timing passes.
- zvec: `IvfRabitqIndexParam(metric_type=IP, nlist=256, total_bits=1..7)`, insert
  and optimize, no refiner. TieredQ regular IVF: 1.16..7.16 code bits/dimension,
  default exact pruning, `fine_tiers=False`. The fractional budgets reproduce
  the prior comparison's approximate size matching; actual bytes are reported.
- Nested IVF: 3..7 total bits/dimension, suffix tier in RAM, prefix 1 bit with
  20 or 50 rerank candidates, or prefix 2 bits with 20 candidates. Both prefix
  index and refinement tier count toward index bytes. `fine_tiers=False`.
- Index size: zvec's quantized `*.qindex.*` files versus TieredQ `storage_bytes`
  (plus the nested tier). These are compressed index accounting figures, not
  measured process RSS. zvec additionally retains a 307,515,392-byte FP32 index
  per collection and collection metadata, excluded from this comparison.
- The fresh TieredQ build passed all 23 selected IVF/nested Python tests; see
  [validation.txt](validation.txt).

## Reproduction and artifacts

The runner imports the existing
[IVF comparison harness](../../experiments/issue6_zvec/ivf_vs_zvec.py), adding
repeated timings and a guard against replacing existing zvec indexes.

Remote workspace: `/home/ubuntu/experiments/tieredq-ivf-zvec-20261106`.
Create a Python 3.12 environment with `uv`, install `./source zvec==0.8.0 pytest`,
then run [job.sh](job.sh) from that workspace. The script expects the exact
source snapshot in `source/`, the three data files in `data/`, and [run.py](run.py)
alongside it. It preserves built zvec collections; use a fresh workspace for
another run. The source snapshot's checksums are in
[source-sha256.txt](source-sha256.txt).

Generate the recall-target comparison with `python analyze.py .` once the TSV
and JSONL result files have been collected here. [analyze.py](analyze.py) selects
the fastest qualifying configuration among those measured, and separately
constrains TieredQ to the fastest qualifying zvec configuration's index size.

This is one dataset and one build per configuration on a shared VM. Timing
passes measure run-to-run variability on those builds; they do not establish
confidence across training seeds or other workloads. End-to-end Python API
timings include zvec's collection overhead. The nested tier is in RAM; SSD
performance is outside this run.

## Follow-up: one million vectors

Continue the 1M-vector comparison on a larger machine. At the follow-up check,
Frieda had about 15 GB of free disk space and 3.2 GiB of available RAM. A million
768-dimensional FP32 vectors alone occupy 2.86 GiB, before index storage and
training buffers, so the current runner has insufficient memory headroom.
Keeping all seven zvec collections would also exceed the available disk space;
a future runner should retain one collection at a time or use more storage.
No 1M-vector run was attempted on Frieda.
