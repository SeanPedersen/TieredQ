# Issue #6: IVF-TieredQ vs RaBitQ IVF (2026-10-30)

Produced by [ivf_eval.cpp](../../experiments/issue6_ivf/ivf_eval.cpp) on Hartmut (i9-7980XE, AVX-512), CPUs 0-7, seed 0.
RaBitQ (official IVF, accurate encoding, high-accuracy scoring) scans TieredQ's partition (same `list_hash`).

Columns: codec budget lists nprobe disk_bits code_bits recall@10 qps_1t_b1 qps_8t_b32 build_s list_hash.

- `baseline-openai100k.tsv`: OpenAI 100k, 64 lists, before the speed work; RaBitQ rows are the reference.
- `final-openai100k-tieredq.tsv`: TieredQ after it, at the same disk sizes. `default` = exact pruning,
  `max4` = `max_bits=4`, `zZ-pP` = `TieredQPruning{Z, P}`.
- `openai1m-lists256-1024.tsv`: OpenAI-1M with coarse lists 256 and 1024, RaBitQ and matched TieredQ
  (measured before adaptive query tiles; single-thread rows are unaffected).
