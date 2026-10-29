# Issue #4 final: TieredQ vs Qdrant TQ+ and RaBitQ at equal measured memory

Question: at each reference codec's **measured** disk, steady-RAM or peak-RAM bits/vector, does TieredQ reach higher Recall@10, and at what speed? References: Qdrant TurboQuant+ (TQ+) and RaBitQ at 1, 2 and 4 bits per dimension. The datasets are SIFT1M (1M × 128, L2), GloVe-200 (100K, cosine), OpenAI3-1536 (100K) and DBpedia Gemini-768 (99K), each with 1,000 queries and seeds 0–2.

The library is `c5657b8` + `83c2a99`:
- chunked covariance
- storage-aware bit allocation
- binary16 fast-cache factor and correction (−32 bits/vector)
- realised budgets for odd bits

The harness is `d9c9c54` + `ee8a9d8`. Host: Hartmut (Intel i9-7980XE, AVX-512), GCC 16.2.1, `-O3 -march=native -ffp-contract=off`.

## Result

**TieredQ wins 35 of 36 targets against TQ+ and 36 of 36 against RaBitQ.** Full tables: [matched.md](matched.md) / [matched.csv](matched.csv). Plots: [recall-memory-qdrant.png](recall-memory-qdrant.png), [recall-memory-rabitq.png](recall-memory-rabitq.png).

Recall@10 margin (TieredQ − reference, percentage points) and TieredQ speed relative to the reference, both at the matched budget:

| Reference | Dataset | Δ R@10 pts | QPS 8t·b32 | QPS 1t·b32 | p50 1t·b1 speedup |
|---|---|---:|---:|---:|---:|
| TQ+ | SIFT | +8.0 … +32.2 | 3.9–11.4× | 4.8–13.1× | 1.1–11.4× |
| TQ+ | GloVe | −0.8 … +4.4 | 5.0–11.5× | 4.9–11.2× | 4.7–10.7× |
| TQ+ | OpenAI | +0.7 … +5.7 | 1.6–6.3× | 2.1–6.9× | 0.71–6.2× |
| TQ+ | Gemini | +1.6 … +11.1 | 1.7–7.1× | 2.4–7.1× | 0.81–7.1× |
| RaBitQ | SIFT | +10.5 … +34.1 | 0.97–1.28× | 1.07–1.52× | 0.21–1.32× |
| RaBitQ | GloVe | +1.0 … +11.7 | 0.85–1.47× | 0.96–1.39× | 0.27–1.38× |
| RaBitQ | OpenAI | +1.1 … +6.0 | 0.32–1.51× | 0.41–1.38× | 0.13–1.30× |
| RaBitQ | Gemini | +1.9 … +9.1 | 0.31–1.58× | 0.38–1.40× | 0.13–1.40× |

- **Against TQ+:**
  - TieredQ is more accurate and faster almost everywhere.
  - The exception is single-query latency at OpenAI and Gemini 4-bit, where TieredQ's p50 is 1.2–1.4× higher (e.g. 13.8 vs 9.8 ms).
  - The one loss is **GloVe 4-bit at equal peak RAM**: 0.8743 vs 0.8826 (−0.8 pts).
  - **GloVe 2-bit at equal peak RAM** is a tie in practice: +0.04 pts, against a TieredQ seed std of 0.35 pts.
- **Against RaBitQ:** TieredQ is more accurate on every target. Speed is mixed:
  - At 1-bit budgets, and on GloVe at 1/2-bit, TieredQ is about as fast or faster.
  - At OpenAI 2/4-bit and Gemini 4-bit, TieredQ is 2.6–3.2× slower batched (8t·b32) and 4.8–7.6× slower single-query (p50 6.1–13.8 ms vs 0.9–1.8 ms). At Gemini 2-bit it is 1.3–1.6× slower on both.
  - Single-query latency is also higher at SIFT 2/4-bit (2.6–4.7×) and GloVe 4-bit (3.3–3.7×).
  - Speeding up these budgets is tracked in #13 and #19.
- Every margin, including the loss, is larger than the seed standard deviations (≤0.005), except the GloVe 2-bit peak tie.

## Method

`benchmark/match_memory.py --quick` runs separately for each reference (`--reference qdrant`, then `rabitq`) on two output directories: `final-match-a` (SIFT, GloVe) and `final-match-b` (OpenAI, Gemini).

1. **Reference cells.** Each reference is measured at 1/2/4 bits and seeds 0–2 with `measure_memory.py`: build in one fresh process, then reload and serve in another.
   - **Disk:** saved index size.
   - **RAM:** steady RSS growth after reload.
   - **Peak:** VmHWM growth over reload plus one untimed pass at threads 1/8 × batch 1/32.
2. **Targets.** Each (reference bits, metric) pair is a target: the minimum of that metric over the reference's seeds.
3. **Calibration.** For each target, find the largest TieredQ `code_bits` whose seed-0 measurement fits. Selection reads storage only, never recall. Every probe is shared across targets and references and logged in `calibration.csv`.
4. **Confirmation.** The chosen budget is measured at seeds 1 and 2. If any seed overshoots, the search steps down to smaller cached budgets, then below them in 8-bit steps (`ee8a9d8`).
5. **Win rule.** A win needs TieredQ's maximum memory over seeds ≤ the target *and* a higher mean recall.
6. **Speed.** [`time_matched.py`](time_matched.py) times TieredQ at every matched budget: seed 0, fresh process per budget, threads 1/8 × batch 1/32, alone on CPUs 0–7.

Reference speed comes from the references' own timing runs at 1/2/4 bits on the same host, inputs and harness. Each row's source is in [reference-timing-provenance.json](reference-timing-provenance.json).
- **TQ+:** [output-qdrant/](output-qdrant/), produced by [run_final.py](run_final.py) on the `source-tq` tree. The Qdrant commit and patch are in `output-qdrant/meta.json`.
- **RaBitQ:** reused from `issue4-range-refresh-20261027`.
- `inputs.json` checks that both used identical inputs.
- TQ+ parallelizes across queries only, so its 8t·b1 schedule scores each single query on one thread.

Compression ratio = 32·d / target bits per vector.

## Corrections during the run

- **Odd-bit crash.** The first pass crashed at GloVe 401 code bits ("tier budget mismatch"). With no 3-bit tiers, an odd leftover bit could not be placed. Fixed in `83c2a99`, which records the realised budget; [run/final-match-a.crash.log](run/final-match-a.crash.log) is the crash log. Cells key timing by the requested budget (`requested_code_bits`).
- **Step-down fix.** The first pass reported "no TieredQ budget fits" at GloVe and Gemini 1-bit peak against TQ+. Peak RSS varies by up to ~20 bits/vector between seeds at 100K vectors, and the matcher only stepped down to budgets it had already measured. With the step-down fix, both became wins:
  - GloVe: 174 code bits, 0.3731 vs 0.3553.
  - Gemini: 634 code bits, 0.7940 vs 0.6980.

  The first-pass results are kept in [first-pass/](first-pass/). No other target changed.
- **Mixed source trees.** Part b's earliest cells were measured on a tree that differed from the final library only by a vectorized binary16 decode (bit-identical values). Storage and recall are therefore identical.

## Library speed A/B

[speed-ab3/](speed-ab3/) is a 4-round interleaved A/B of the final library against the pre-change build (`source-build`). It covers OpenAI 2/4-bit and Gemini 2-bit, with a fresh process per cell, on CPUs 0–7.
- Median QPS is within noise or better on all schedules (8t·b32: +0.4 to +5.9%).
- Recall is identical.
- The library changes also cut build peak RSS by 50–55% (OpenAI 1-bit 598 → 267 MiB) and build time by 10–14%.

## Files

- `final-match-{a,b}/`: `cells.csv` (every measured cell), `calibration.csv`, `matches-{qdrant,rabitq}.{json,md}`, `meta-*.json`, and per-cell `build.json`/`serve.json`.
- `output-final-matched-timing/`: TieredQ timing at every matched budget.
- `output-qdrant/`, `output-memory-qdrant/`: TQ+ timing (1/2/4 bits, seeds 0–2) and TQ+ memory cells (quick mode), from which the TQ+ reference cells were seeded ([seed_refs.py](seed_refs.py)).
- `run/`: job scripts, logs, host process snapshots, UTC timestamps and `source-commit.txt`.
- [finalize.py](finalize.py): audits inputs, the reference timing grid and the matched grid. It also checks that TieredQ's timed seed-0 recall equals its match-cell recall at every budget. It writes the tables and plots.

## Limitations

- **Fixed costs at 100K.** Fixed per-process costs are divided by N; at larger N, every codec's overhead per vector shrinks.
- **Peak-RAM noise.** Peak RSS differs by up to ~20 bits/vector between seeds, and seeds 1/2 only confirm budgets. Peak-RAM margins are therefore less precise than disk margins, which are deterministic.
- **Narrow budgets for TieredQ's speed.** TieredQ's speed is measured at one matched budget per target. The reference's speed is at its own nominal bits.
- **No 3-bit references.** Fast mode currently allocates no 3-bit tiers when all tiers are ≤4 bits; #20 tracks dense 3-bit scoring, which could help the GloVe/SIFT matches near 2.5–3.8 bits/dim.
