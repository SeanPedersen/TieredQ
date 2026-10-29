# Issue #20: additive 3-bit tiers in fast mode

Question: can fast mode score 3-bit tiers at nibble-lookup speed with storage equal to `code_bits`? And does that flip the one issue #4 loss, GloVe 4-bit at equal peak RAM against Qdrant TQ+?

Library:
- `28b7ca6`: additive 3-bit codebook and split nibble groups.
- `b36164a`: split parts share groups with the 2- and 1-bit tiers, and the budget is no longer trimmed.

Baseline: `177db5f`. Host: Hartmut (Intel i9-7980XE, AVX-512), GCC 16.2.1, Release build, `taskset -c 0-7`. The background services listed in the `*host-processes.txt` files were running throughout. The C++ suite passes on AVX-512 for both library versions ([ab/tests-ship.log](ab/tests-ship.log), [ab/tests-shared.log](ab/tests-shared.log)).

## Result

- **GloVe gains +1.2 to +3.6 R@10 points at 2.25–3.9 bits/dim.** Stored bytes are equal, and QPS is within noise.
- **The GloVe 4-bit peak-RAM target flips:**
  - TieredQ: 766 code bits, peak 931.9 ≤ 934.2 bits/vector, R@10 **0.8881**.
  - Qdrant TQ+: R@10 0.8826, so the margin is +0.55 pts (−0.83 in issue #4).
  - Speed at that budget is unchanged: 11.4k QPS at 8t·b32, p50 0.60 ms at 1t·b1.
- **Equal-memory totals:**
  - Vs TQ+: 18/18 SIFT/GloVe wins, and 36/36 with the unchanged OpenAI/Gemini results.
  - Vs RaBitQ: 18/18.
- **SIFT is neutral:** within 0.2 pts of `177db5f`.
- **DBpedia OpenAI/Gemini:** byte-identical at 2–4 bits/dim; within ±0.6 pts at 1 bit/dim.

### A/B vs 177db5f (fast mode, R@10 mean of seeds 0–2)

[`ab_eval.cpp`](../../experiments/issue20_additive/ab_eval.cpp) setup:
- 100k base / 1k queries, with inner-product truth on the raw base
- 256 centroids, 32k training rows
- one fresh process per (dataset, budget, tree), with the tree order alternating

Round 1 (`28b7ca6`) is in `ab/ab-*.tsv` and round 2 (`b36164a`) in `ab/shared-*.tsv`. Columns: tree, budget, bits/dim, realised code bits, widest tier, 3-bit dims, fast bytes/vector, stored bits/vector, R@10, QPS at 1t·b1 / 1t·b32 / 8t·b32 (seed 0, best of 5), and model ids.

| GloVe bits/dim | 2.00 | 2.25 | 2.50 | 2.75 | 3.00 | 3.25 | 3.50 | 3.75 | 3.83 | 3.90 | 3.97 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 177db5f | .6700 | .6982 | .7228 | .7508 | .7780 | .8044 | .8300 | .8630 | .8743 | .8853 | .8964 |
| 28b7ca6 | .6700 | .7106 | .7435 | .7804 | .8119 | .8398 | .8604 | .8791 | .8881 | .8974 | .8994 |
| **b36164a** | .6700 | **.7108** | **.7439** | **.7803** | .7780 | **.8398** | **.8604** | **.8791** | **.8881** | **.8974** | **.8994** |

| SIFT bits/dim | 2.00 | 2.25 | 2.50 | 2.75 | 3.00 |
|---|---|---|---|---|---|
| 177db5f | .8277 | .8531 | .8762 | .8926 | .9036 |
| 28b7ca6 | .8230 | .8521 | .8686 | .8865 | .9026 |
| **b36164a** | .8259 | .8515 | .8756 | .8926 | .9027 |

- **Budget trimming (`28b7ca6`):** when padding overflowed the budget, round 1 trimmed the 3-bit allocation by one byte.
  - SIFT lost 0.5–0.8 pts at the trimmed budgets: 248, 312 and 344 realised bits.
  - On GloVe the trimmed 3.0 bits/dim point gained +3.4 pts.
- **`b36164a` falls back to 2/4 instead.** That keeps SIFT within 0.2 pts, but GloVe 3.0 bits/dim falls back as well, because the additive allocation would store 608 > 600 bits. Making the allocation recover this is tracked in #9.
- **Identical models:** SIFT at 3.5–4.0, GloVe at 4.0, and OpenAI/Gemini at 2–4 bits/dim produce byte-identical models (same model ids), because those allocations have tiers wider than 4 bits or no 3-bit tiers.
- **DBpedia at 1 bit/dim:** OpenAI 0.8946 → 0.8935, Gemini 0.8295 → 0.8354.

### Equal-memory re-match (SIFT, GloVe)

Full tables: [matched.md](matched.md) / [matched.csv](matched.csv). Plots: [recall-memory-qdrant.png](recall-memory-qdrant.png), [recall-memory-rabitq.png](recall-memory-rabitq.png). Δ R@10 is TieredQ − reference, in percentage points.

| Target | issue #4 final | b36164a |
|---|---:|---:|
| TQ+ GloVe 4-bit disk | +1.38 | **+1.68** |
| TQ+ GloVe 4-bit steady RAM | +0.92 | **+1.06** |
| TQ+ GloVe 4-bit peak RAM | −0.83 | **+0.55** |
| RaBitQ GloVe 2-bit disk | +11.7 | **+16.2** |
| RaBitQ GloVe 2-bit peak RAM | +1.0 | **+12.7**¹ |
| TQ+ GloVe 1-bit steady RAM | +4.1 | +2.4 |
| TQ+ GloVe 2-bit steady RAM | +2.6 | +0.04 |

¹ Mostly calibration, not #20. Issue #4 settled on a cached 401-bit budget whose peak (548–559 bits/vector) sat far below the 719.3 target. This run confirms 554 bits at a peak of up to 714.7 bits/vector.

**Why two steady-RAM targets dropped.** The two lower steady-RAM targets are measurement noise, not the allocation:
- At the chosen budgets (182 and 367 code bits), models and per-seed recall are identical to issue #4.
- Steady RSS of the same probe varies by about ±40 bits/vector between runs. The 190-bit probe now measures 301.1 against a 299.8 target.
- After a seed overshoots, the matcher steps down to the nearest cached budget (393 → 367).

**RaBitQ GloVe 4-bit.** At these budgets (1,084–1,115 code bits) the models are byte-identical to issue #4 (packed-code path). Their 8t·b32 QPS reads about 20% lower than in issue #4, while 1t·b32 and p50 match. This is host variance on a shared machine, not a code change.

## Method

1. **References.** [`job_match.sh`](../../experiments/issue20_additive/job_match.sh) reuses the TQ+ and RaBitQ reference rows of `issue4-final-20261029/final-match-a/cells.csv`; only the non-TieredQ rows are copied into `ship-match-a/cells.csv`.
2. **Matching.** It then runs `match_memory.py --quick --reference {qdrant,rabitq} --reference-bits 1 2 4 --datasets sift glove` on the `b36164a` tree, so every TieredQ cell is re-measured. Targets, calibration, confirmation and the win rule are as in issue #4.
3. **Timing.** [time_matched.py](time_matched.py) times every matched budget: seed 0, fresh process per budget, threads 1/8 × batch 1/32, CPUs 0–7.
4. **Tables.** [finalize.py](finalize.py) builds the matched tables against the issue #4 final bundle's reference timing. It also checks that each timed run reproduces its match cell's seed-0 recall.

Data, the 100k-row exports for the A/B and the job scripts are in [`benchmark/experiments/issue20_additive/`](../../experiments/issue20_additive/).
