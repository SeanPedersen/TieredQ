# Issue #22 phase 1: MS MARCO-1M, real vs document queries, IVF + rerank (2026-10-31)

Data: [msmarco1m_data.py](../../msmarco1m_data.py) (`manifest.json`): Cohere embed-english-v3.0 (1,024-d), 1,000,000
passages drawn with seed 42 from the first two corpus shards. Two query sets on that base: `real`, 1,000 MS MARCO dev
questions (seed 42), and `doc`, 1,000 held-out passages from the same draw. Exact FP32 inner-product truth.
Runs: [ivf_eval.cpp](../../experiments/issue6_ivf/ivf_eval.cpp) via `job_rerank.sh` / `job_native_time.sh` (`DIM=1024`),
256 coarse lists shared by both codecs, Hartmut (i9-7980XE); recall passes on CPUs 8-31, timing alone on CPUs 0-7.

| Path | Contents |
|---|---|
| `real-recall/`, `doc-recall/`, `*-recall-tables.md` | untimed: TieredQ prefixes 256-1024 x fp32 / int8 / refinement:3,4; RaBitQ 1 x fp32 / int8 / rabitq:5,6; nprobe 8-128, k' 20-500 |
| `real-time/`, `real-time-tables.md` | real queries timed: cells selected per target, plus RaBitQ native 5 / 6 / 7 bit (`native-256-*`) |

## Findings
**The #22 gate passes at about 1 bit/dim.** First-stage candidate recall (true top 10 among k') against RaBitQ 1 bit
(1,165 index bits/vector):

| TieredQ | Index bits | nprobe, k' | real | delta real | doc | delta doc |
|---|---|---|---|---|---|---|
| 1024 | 1,133 | 128, 20 | 0.977 | +3.4 pt | 0.991 | +2.2 pt |
| 1024 | 1,133 | 32, 20 | 0.925 | +2.6 pt | 0.975 | +2.2 pt |
| 768 | 867 | 128, 20 | 0.945 | +0.2 pt | 0.978 | +0.8 pt |
| 512 | 608 | 128, 20 | 0.877 | -6.6 pt | 0.938 | -3.1 pt |

At equal index RAM the gain is larger on real queries; 768 (-26% RAM) ties RaBitQ on both sets. Sub-1-bit savings
shrink on real queries: 512 needs about 4x k' (2x on doc). Real questions are also harder for the IVF itself: both
codecs need about 4x nprobe for the doc-query recall, and FP32 rerank reaches 0.99 only at nprobe 128, k' >= 100.

**Real queries, timed** (`real-time-tables.md`; total bits = index + rerank tier per vector, 1t QPS):

| R@10 | TieredQ | Total bits | Recall | QPS | RaBitQ | Total bits | Recall | QPS | Speed |
|---|---|---|---|---|---|---|---|---|---|
| 0.90 | 512 + refinement:3, nprobe 32, k' 100 | 3,224 | 0.901 | 902 | native 5 bit, nprobe 32 | 5,325 | 0.921 | 519 | 1.74x |
| 0.95 | 512 + refinement:4, nprobe 64, k' 500 | 4,248 | 0.952 | 439 | native 6 bit, nprobe 64 | 6,349 | 0.958 | 266 | 1.65x |
| 0.95 | 768 + refinement:4, nprobe 64, k' 100 | 4,139 | 0.952 | 361 | native 6 bit, nprobe 64 | 6,349 | 0.958 | 266 | 1.36x |
| 0.97 | 512 + refinement:4, nprobe 128, k' 500 | 4,248 | 0.972 | 243 | native 6 bit, nprobe 128 | 6,349 | 0.978 | 134 | 1.81x |
| 0.97 | 1024 + refinement:4, nprobe 128, k' 50 | 4,277 | 0.975 | 160 | native 6 bit, nprobe 128 | 6,349 | 0.978 | 134 | 1.19x |
| 0.95 | 512 + INT8, nprobe 64, k' 200 | 8,800 | 0.957 | 526 | 1 bit + INT8, nprobe 64, k' 50 | 9,357 | 0.957 | 263 | 2.0x |
| 0.97 | 512 + INT8, nprobe 128, k' 200 | 8,800 | 0.977 | 284 | 1 bit + INT8, nprobe 128, k' 50 | 9,357 | 0.978 | 132 | 2.15x |

- Nested TieredQ vs RaBitQ's native compressed IVF: 1.2-1.8x QPS at 33-42% less total memory per vector.
- With an INT8 tier: 2.0-2.15x QPS at -48% index RAM (608 vs 1,165 bits), the tier read by id for k' candidates.
- The 512 prefix wins on speed despite the larger k' it needs: its scan is cheap and rescoring k' = 200-500 is not.
- Refinement recall tops out at 0.972-0.975 on real queries (0.982-0.985 on doc); 0.98+ needs INT8 / FP32.
- RaBitQ's native IVF and the `rabitq:B` rescore glue run at the same speed here.

Next for #22: query-aware allocation to recover the 512 prefix's real-query loss, the equal-memory flat match and the
alpha run on both query sets, and the cost model.
