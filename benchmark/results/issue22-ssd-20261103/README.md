# Issue #22: TieredQ nested IVF with an SSD tier against RaBitQ's native IVF in RAM (2026-11-03)

OpenAI-1M (DBpedia, 1,536-d), 256 lists from TieredQ's k-means and shared with RaBitQ, `issue6_ivf_eval` at 6ef5c0a
(`SOURCE_REVISION`), seed 0.
- **RaBitQ:** native IVF; its B-bit code is in RAM and it reads nothing from disk.
- **TieredQ:** `TieredQNestedIVFIndex`; the prefix IVF is in RAM. The `prefix_and_suffix` tier (4 bits/dim total) is a
  file on the home SSD (Samsung 960 EVO, LUKS + btrfs), read with direct I/O (no page cache) with 32 reads in flight
  per query (io_uring).
- **Phases:**
  - `job_recall.sh`: recall sweep, untimed, CPUs 8-31 (`recall.tsv`).
  - `job_time.sh`: equal-recall configs timed alone on CPUs 0-7, 2 interleaved rounds, median (`time.tsv`). RaBitQ
    5-bit at nprobe 32 was timed afterwards with the same command.
- `summary.py time.tsv` prints `tables.md`.

## At equal recall: RaBitQ's fastest config against TieredQ
RAM is the IVF index per vector; SSD is the tier row per vector; reads are k′ per query.

| R@10 level | Config | RAM bits/vector | SSD B/vector | Reads/query | R@10 | QPS 1t·b1 | QPS 8t·b32 |
|---|---|---:|---:|---:|---:|---:|---:|
| ~0.955 | RaBitQ 5-bit, nprobe 32 | 7,891 | - | 0 | 0.9589 | 340 | 2,404 |
| | TieredQ p768, nprobe 32, k′ 50 | 876 (0.11x) | 770 | 50 | 0.9554 | 330 (0.97x) | 1,731 (0.72x) |
| ~0.972 | RaBitQ 5-bit, nprobe 64 | 7,891 | - | 0 | 0.9755 | 174 | 1,252 |
| | TieredQ p768, nprobe 64, k′ 50 | 876 (0.11x) | 770 | 50 | 0.9715 | 212 (1.22x) | 1,198 (0.96x) |
| ~0.981 | RaBitQ 6-bit, nprobe 64 | 9,427 | - | 0 | 0.9811 | 176 | 1,248 |
| | TieredQ p512, nprobe 128, k′ 100 | 616 (0.065x) | 790 | 100 | 0.9810 | 162 (0.92x) | 962 (0.77x) |

Against RaBitQ's lowest-RAM config at each level (3-bit / 4-bit / 5-bit at nprobe 128, 4,819-7,891 bits), TieredQ is
1.8-3.7x faster at 1 thread and 1.5-2.7x at 8 threads (`tables.md`).

## Findings
- With the tier on SSD, TieredQ keeps 6.5-11% of RaBitQ's RAM at equal recall.
- Single-thread QPS is on par: 0.92-1.22x. At 8 threads TieredQ is at 0.72-0.96x of RaBitQ's fastest config.
- The 8-thread runs read 60-96k rows/s, below the ~160k/s this SSD sustained in #23. So these cells are compute-bound,
  not IOPS-bound. On a faster CPU, or with more threads, the IOPS cap `~160k / k′` would bind first.
- RaBitQ's fastest config at each level uses 5-6 bits and fewer probes. Its lowest-RAM config (3-4 bits, nprobe 128)
  is 2-4x slower than its fastest one.
- One seed, one host, one SSD. Recall at seeds 1-2 and a second device are not measured.
