# Issues #19.1 / #22 / #23: IVF + rerank, sub-1-bit prefixes and nested rerank codes (2026-10-30)

Produced by [ivf_eval.cpp](../../experiments/issue6_ivf/ivf_eval.cpp) through
[job_rerank.sh](../../experiments/issue6_ivf/job_rerank.sh) on Hartmut (i9-7980XE, AVX-512), seed 0. RaBitQ (official IVF,
accurate encoding, high-accuracy scoring) scans TieredQ's partition (same `list_hash`). Recall passes ran on CPUs 8-31;
timing ran alone on CPUs 0-7 (QPS 1t = one query at a time, 8t = 8 threads x batches of 32, both including rescoring).
Queries are held-out DBpedia documents (inner-product truth top 10, 1,000 queries).

Rerank rows: label codec budget lists nprobe disk_bits code_bits rerank_k cand_recall recall reads fp32_bits qps_1t qps_8t
build_s list_hash [rescore rescore_bytes]. `disk_bits` is the serialized IVF (index RAM) per vector; `rescore` is the
rerank tier (`fp32`, `fp16`, `int8`, `tieredq:B` flat, `tieredq-residual:B`, `tieredq-nested:B` = eval prototype,
`tieredq-refinement:B` = library `TieredQRefinement`, `rabitq:B` = RaBitQ's 1-bit code plus B-1 ex-bits) and
`rescore_bytes` its bytes per vector (suffix only for residual / nested / refinement). Tables come from
[report_rerank.py](../../experiments/issue6_ivf/report_rerank.py) (`report` for timed dirs, `rescore` for rescore dirs).

## Files
| Path | Content |
|---|---|
| `openai100k-recall/`, `openai100k-time/`, `openai100k-tables.md` | OpenAI 100k, 64 lists: TieredQ at RaBitQ's 1/2/4-bit disk sizes vs RaBitQ, exact FP32 rerank; untimed sweep and the timed cells `select` picked |
| `openai100k-rescore/` | 1416 and RaBitQ-1 first stages, every rerank tier, nprobe 8-64, k' 20-200; `native-0-B.tsv` = RaBitQ's native B-bit IVF without rerank (no-rerank row format); `nesting.txt` = `tieredq-nesting 1416,2876,5820` |
| `openai100k-residual/` | residual tiers (TieredQ on x - decode(prefix)) |
| `openai100k-prefix-prototype/` | prefixes 256 / 512 / 768 / 1024 / 1416 with FP32 and prototype nested tiers, k' up to 1000 |
| `openai100k-rescore-tables.md` | per first stage and target, each tier's first (nprobe, k') cell |
| `openai100k-rescore-time/` | timed: prefixes x fp32 / int8 / refinement (per-position LUT scorer), `z8-p3` prefix pruning, RaBitQ-1 x fp32 / int8 / rabitq:5,6 |
| `openai100k-rescore-time2/` | timed: refinement with the shared level-table scorer, int8 anchor in the same process |
| `openai1m-recall/`, `openai1m-time/`, `openai1m-tables.md` | OpenAI-1M (999k), coarse lists 256 and 1024, matched budgets, FP32 rerank |
| `openai100k-rescore-time3/` | timed: the time2 cells with this commit's scorer (gather-free AVX-512 fields, int8 query panels, the IVF's query weights shared) |
| `openai1m-rescore-recall/`, `openai1m-rescore-recall-tables.md` | OpenAI-1M, 256 lists: prefixes 512 / 768 / 1024 x fp32 / int8 / refinement:3,4 and RaBitQ 1 x int8 / rabitq:5,6, untimed |
| `native-time/` | RaBitQ's native IVF at 5 / 6 / 7 bits (1-bit scan plus ex-bit refinement, whole code in RAM, no rerank), timed: `native-64-*` 100k, `native-256-*` 1M ([job_native_time.sh](../../experiments/issue6_ivf/job_native_time.sh)) |
| `openai1m-rescore-time/`, `openai1m-rescore-time2/`, `openai1m-rescore-tables.md` | the selected 1M cells timed; the refinement rows of `openai1m-rescore-time/` ran on the default 1416 prefix (a `select` bug, fixed) and are void, `time2` re-times them |

## Findings
**Sub-1-bit TieredQ IVF + INT8 rerank beats RaBitQ 1-bit + INT8 on index RAM and QPS** (100k, 1t QPS):

| R@10 | Config | Index bits/vec | Recall | QPS 1t / 8t |
|---|---|---|---|---|
| ≈0.97 | RaBitQ 1 bit, nprobe 16, k' 50 | 1,711 | 0.973 | 1,931 / 14,042 |
| ≈0.97 | TieredQ 768, nprobe 16, k' 50 | 976 | 0.970 | 3,778 / 28,273 |
| ≈0.99 | RaBitQ 1 bit, nprobe 32, k' 50 | 1,711 | 0.990 | 1,078 / 7,728 |
| ≈0.99 | TieredQ 1024, nprobe 32, k' 50 | 1,262 | 0.990 | 1,974 / 13,874 |
| ≈0.99 | TieredQ 512, nprobe 32, k' 100 | 690 | 0.988 | 2,704 / 18,538 |

**Nested rerank codes** (`TieredQRefinement`) cost about no recall against flat codes of the full width (1416 prefix:
0.987 vs 0.987 at 4 bits/dim, 0.992 vs 0.993 at 5) and need fewer bytes than RaBitQ's ex-bits at equal recall
(399 / 599 / 769 vs 584 / 776 / 968 B). Against RaBitQ 1 + its 1->6 codes: 512 + refinement:4 at 0.95 gives -60%
index RAM, -39% tier bytes, 1.4x QPS; 1024 + refinement:4 at 0.98 gives -26% / -44% / 1.36x (0.981 vs 0.985).
(The QPS here are the earlier scorer's; see below.)
`rabitq:B` rescoring is generic eval glue: its recall matches the native IVF, its speed may understate RaBitQ.

**Dead ends:** residual tiers gain nothing from the prefix (equal to a flat code one bit lower); separately trained
models are not refinements (tiers do not line up, `nesting.txt`); `z8-p3` pruning of the 1416 prefix does not speed it up.

**Faster refinement scoring** (100k, `openai100k-rescore-time3/` vs `time2`, same cells): 1.5-1.9x the 1t QPS at
unchanged recall, now 0.6-0.9x of INT8 rescoring on the same candidates (e.g. 1024 + refinement:4, nprobe 32, k' 50:
1,599 vs INT8 1,956; was 1,051). Rescoring alone at 768, k' 50 went from about 440 to 136 us per query (INT8: 23 us):
the scan's query weights are reused, fields are unpacked with shuffles instead of gathers (about 0.75 us per candidate,
was 4) and the query projects onto int8 panels in one sequential stream. What is left is that projection: the finer
model codes about 1,530 of 1,536 ranks, so a query is projected onto about 1,140 suffix-only ranks (about d^2 / 1.4
multiply-adds, memory-bound) against k' x d for INT8. Closing it needs a codec change (e.g. a fast structured rotation
for the tail ranks) or batched queries sharing the panels.
At 0.95: 512 + refinement:4 (nprobe 16, k' 50) gives 2,929 / 21,261 QPS vs RaBitQ 1 + INT8 2,059 / 14,331 and
RaBitQ 1 + 1->6 1,215 / 9,181. At 0.98: 1024 + refinement:4 (nprobe 32, k' 50, 0.981) gives 1,599 / 10,893 vs
RaBitQ 1 + 1->6 (0.985) 774 / 5,860.

**1M confirms it** (`openai1m-rescore-tables.md`, 256 lists, 1t QPS). Index RAM: TieredQ 512 / 768 / 1024 use 627 /
886 / 1,138 bits/vector vs RaBitQ's 1,683.
- INT8 rerank: 512 reaches 0.95 / 0.98 / 0.99 at 914 / 520 / 271 QPS vs RaBitQ 1 + INT8 332 / 168 / 87 (2.8-3.1x,
  -63% index RAM). 1024 matches RaBitQ's cells at -32% index RAM and 1.6x QPS.
- Refinement vs RaBitQ's own codes: 0.95: 512 + refinement:4 (702 B) 719 QPS vs rabitq:5 (980 B) 298; 0.97: 768 +
  refinement:4 (672 B) 339 vs 162; 0.98: 768 + refinement:4 at nprobe 128 187 vs rabitq:6 (1,172 B) 157.
- Refinement tops out at 0.983-0.984 at nprobe 128; 0.99 needs INT8. The 512 prefix needs k' 100-200 with refinement.

**Against RaBitQ's native B-bit IVF** (`native-time/`; its scan refines with the ex-bits in place, the fastest form of
its compressed codes). Total memory = index bits + rerank-tier bits per vector; 1t QPS:

| Set | R@10 | TieredQ | Total bits | Recall | QPS | RaBitQ native | Total bits | Recall | QPS | Speed |
|---|---|---|---|---|---|---|---|---|---|---|
| 100k | 0.95 | 512 + refinement:4, nprobe 16, k' 50 | 6,378 | 0.953 | 2,929 | 5 bit, nprobe 16 | 7,919 | 0.962 | 1,895 | 1.55x |
| 100k | 0.97 | 512 + refinement:4, nprobe 32, k' 100 | 6,378 | 0.976 | 1,980 | 6 bit, nprobe 16 | 9,455 | 0.968 | 1,799 | 1.10x |
| 100k | 0.98 | 1024 + refinement:4, nprobe 32, k' 50 | 6,526 | 0.981 | 1,599 | 6 bit, nprobe 32 | 9,455 | 0.985 | 981 | 1.63x |
| 1M | 0.95 | 512 + refinement:4, nprobe 32, k' 100 | 6,243 | 0.954 | 719 | 5 bit, nprobe 32 | 7,891 | 0.963 | 331 | 2.2x |
| 1M | 0.97 | 768 + refinement:4, nprobe 64, k' 50 | 6,262 | 0.973 | 339 | 6 bit, nprobe 64 | 9,427 | 0.982 | 174 | 1.95x |
| 1M | 0.98 | 768 + refinement:4, nprobe 128, k' 50 | 6,262 | 0.980 | 187 | 6 bit, nprobe 64 | 9,427 | 0.982 | 174 | 1.07x |
| 1M | 0.99 | 512 + INT8, nprobe 128, k' 100 | 12,915 | 0.990 | 271 | 6 bit, nprobe 128 | 9,427 | 0.990 | 88 | 3.1x |

The native IVF is 1.6x faster than the `rabitq:B` rescore glue at 100k and about equal at 1M; TieredQ keeps 1.1-2.2x at
20-34% less total memory up to 0.98 (R@10 0.98 at 1M is the narrowest), and above that INT8 rerank is 3.1x faster
with more total bits (the INT8 tier can live on SSD; RaBitQ's codes are the scanned index).

**1M, FP32 rerank at 1-bit index RAM:** TieredQ 1558 vs RaBitQ 1 bit is 1.17x faster at 0.95 and 1.23x at 0.99 with 256
lists, 0.85-0.94x with 1024 lists (whose nprobe <= 128 caps recall at 0.981). More index bits add no speed once rerank is
there.

Discussion: #23 (nested codes), #22 (cost model), #19 (adoption).
