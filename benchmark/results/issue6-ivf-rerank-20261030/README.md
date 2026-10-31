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
Refinement rescoring still runs at about 0.5x INT8 rescoring QPS (about 0.24 ms per query + 4 us per candidate).
`rabitq:B` rescoring is generic eval glue: its recall matches the native IVF, its speed may understate RaBitQ.

**Dead ends:** residual tiers gain nothing from the prefix (equal to a flat code one bit lower); separately trained
models are not refinements (tiers do not line up, `nesting.txt`); `z8-p3` pruning of the 1416 prefix does not speed it up.

**1M, FP32 rerank at 1-bit index RAM:** TieredQ 1558 vs RaBitQ 1 bit is 1.17x faster at 0.95 and 1.23x at 0.99 with 256
lists, 0.85-0.94x with 1024 lists (whose nprobe <= 128 caps recall at 0.981). More index bits add no speed once rerank is
there. Sub-1-bit prefixes and refinement are not yet measured at 1M.

Discussion: #23 (nested codes), #22 (cost model), #19 (adoption).
