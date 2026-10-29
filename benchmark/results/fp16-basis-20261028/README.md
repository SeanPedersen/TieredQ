# FP16 PCA basis (model format 3)

Issue #4 follow-up: after the exact serving-RAM fixes, the PCA basis (d × m FP32) was TieredQ's largest remaining fixed cost: 418 bits/vector on OpenAI 1-bit at 100K and 677 at 2-bit. This bundle covers the model-knob exploration and the FP16 basis that is now the default.

Hartmut (Intel i9-7980XE, AVX-512), GCC 16.2.1, `-O3 -march=native -ffp-contract=off`. Exploration ran on CPUs 8–15.

## Exploration

- **Fewer centroids** ([knob-centroids.jsonl](knob-centroids.jsonl), [knob_centroids.py](knob_centroids.py)): k = 32/64/128 instead of 256 saves up to about 115 bits/vector on OpenAI. Spending that saving on code bits changes recall@10 by −0.2 to +0.6 points at matched accounted storage (seed 0), which is within noise, and Gemini behaves the same. Not adopted.
- **FP16 basis** ([knob-fp16.jsonl](knob-fp16.jsonl), [knob_fp16.py](knob_fp16.py)): rounding the trained basis to binary16 changed recall@10 by −0.0011 to +0.0007 (mean −0.0002) over 24 dataset × budget × seed cells. Adopted.

## Change

Training rounds the basis once to IEEE binary16 (round-to-nearest-even) and fits the projection statistics to the rounded values, so serving and encoding use exactly what is persisted. Model format 3 stores the binary16 basis; RAM holds only that copy. Query projection widens it in registers (`cvtph`) with the same per-element multiply-then-add as FP32. A new test checks that a format-3 model encodes and searches (exact and fast, lookup and level layouts) bit-identically to a format-2 model holding the same rounded basis in FP32. Format-2 files still load, serve unchanged and re-serialize to identical bytes, and older readers reject format 3.

## Result

[storage-fp32.jsonl](storage-fp32.jsonl) and [storage-fp16.jsonl](storage-fp16.jsonl) cover seed-0 indexes at 100K vectors (Gemini 99K), measured with [compare_storage.py](compare_storage.py). RAM is allocation accounting of the reloaded index.

| Cell | Disk and RAM saved (bits/vector) | Recall@10 FP32 → FP16 |
|---|---:|---|
| OpenAI 1 / 2 / 4-bit | −209 / −339 / −377 | 0.8939 → 0.8933, 0.9474 → 0.9470, 0.9839 → 0.9847 |
| Gemini 1 / 2 / 4-bit | −51 / −77 / −95 | 0.8337 → 0.8337, 0.9252 → 0.9251, 0.9784 → 0.9779 |
| GloVe 1 / 2 / 4-bit | −6 / −6 / −6 | 0.4121 → 0.4110, 0.6719 → 0.6717, 0.9011 → 0.9009 |

## Speed

Because the format changes, each tree serves its own index built with the same seed and configuration; the old library cannot read format 3. The reference tree is the previous HEAD (`d928b7e` library).

- **First version** ([ab-fp16-scratch-row-table.md](ab-fp16-scratch-row-table.md)). It widened each basis row into a scratch buffer per 8-query tile.
  - 1 thread, 6–7 rounds: 17 of 18 cells within noise; lookup layouts were unchanged or faster (OpenAI 1-bit t1-b1 +7.5%).
  - 8 threads, 2–3 rounds: OpenAI 4-bit t8-b32 −7%, Gemini 2-bit −4–6%.
  - This run shared the host with a memory job and is noisier.
- **Final version** ([ab-fp16b-table.md](ab-fp16b-table.md), [ab-fp16b.jsonl](ab-fp16b.jsonl); 6 interleaved rounds on quiet CPUs 0–7). Widening now happens in registers inside the multiply loop, with the same per-element arithmetic. Tested on the level layouts the change touches (OpenAI 2/4-bit, Gemini 2-bit):

| Index | t1-b1 QPS | t1-b32 QPS | t8-b32 QPS (information) |
|---|---:|---:|---:|
| OpenAI 2-bit | 1.011× | 1.021× | 1.042× |
| OpenAI 4-bit | 1.017× | 1.004× | 1.033× |
| Gemini 2-bit | 0.998× | 0.986× | 1.023× |

Every 1-thread cell is within the reference tree's own spread (≤2.4%) or faster, with p50 ratios of 0.98–1.003, and the 8-thread slowdown is gone. The lookup-layout weight kernels did not change after the first A/B.

## Limitations

These are fixed per-model costs, divided by N = 100K. Recall deltas are single-seed for the storage table; the exploration covers seeds 0–2.
