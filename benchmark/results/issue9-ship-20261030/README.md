# Issue #9 ship verification (Hartmut, 2026-10-29/30)

Exact padded allocation on variance^1.5 as the default, native metrics and the AVX-512 wide-encoder change, against
c85b0f8 (base). Trees: `ship` = 85887a4 + the `AB_MODE` eval, `final` = f5aebb9 (native metrics, encoder change).
Job scripts: `benchmark/experiments/issue9_empirical/job_ship.sh` (fast/full recall, first re-match) and
`job_final.sh` (SIFT re-match with the native L2 metric, timing, matched timing). Host: i9-7980XE, 36 CPUs, shared.

## Recall A/B (100k base, 1k queries, inner-product truth, seeds 0-2)
- `fast/ab-*.tsv`: `base` (c85b0f8, copied from the #9 held-out run), `layout-p1.5` (the #9 experiment) and `ship`.
  `ship` rows equal `layout-p1.5` rows exactly. Held-out (odd queries) Δ R@10: GloVe +0.55, SIFT +0.93,
  OpenAI +0.60, Gemini +0.79, pooled +0.72 pt (`heldout.py fast`).
- `full/ab-*.tsv`: full mode on variance (`full-p1`) vs variance^1.5 (`full-p1.5`): +0.04 / +1.04 / +0.68 / +0.71 pt
  mean (GloVe / SIFT / OpenAI / Gemini), worst budget -0.05. Shipped as the full-mode default in 1ab1c9b.

## Equal-memory re-match vs RaBitQ and Qdrant TQ+ (`match_memory.py --quick`)
Reference cells are reused from the #20 (SIFT/GloVe) and #4 final (OpenAI/Gemini) matches; TieredQ cells are new.
Compare with `compare_matches.py OLD:NEW` (Δ R@10 of new vs old TieredQ at equal measured memory).
- `match-b/` (OpenAI, Gemini; ship tree): every disk and RAM target gains, +0.28 to +1.15 pt.
- `match-a/` (ship tree): GloVe disk targets +1.32 / -0.13 / 0.00 pt vs RaBitQ and +0.43 / -0.14 / -0.15 vs Qdrant
  at 1 / 2 / 4 bits/dim; its RAM targets swing (-1.6 to +4.3) with calibration picking other budgets. Its SIFT rows
  lose 1.5-5 pt: the benchmark's SIFT L2 lift whitens queries, for which variance^1.5 is the wrong weighting.
  `diag-p1/` (power 1 tree) reproduces the base SIFT cells exactly.
- `match-sift/` (final tree, native L2 metric, power 1): -0.20 to +0.34 pt. `diag-metric/` shows native L2 matching
  the old lift at the base budgets (R@10 0.5902 / 0.8532 / 0.9599 vs 0.5894 / 0.8501 / 0.9615).
- Pooled over SIFT (native), OpenAI and Gemini disk targets: +0.47 pt vs Qdrant TQ+, +0.53 pt vs RaBitQ.
  TieredQ still beats both references in every cell. Peak-RAM targets jump with calibration noise; use disk/RAM.

## Cost
- `final-time-*.tsv` (`issue9_build_time`, 100k base, 3 interleaved rounds on CPUs 0-7, base / ship / final):
  training unchanged (-8% to +0%); add within ±5% of base except GloVe 3.0 bits/dim (+27%, additive 3-bit tiers) and
  OpenAI 1.0 (-12%); the encoder change brings high-bit add back to base (Gemini 4.0: 1.03 / 1.32 / 1.08 s,
  OpenAI 4.0: 2.56 / 2.95 / 2.52 s). Search QPS: +13% to +35% on OpenAI/Gemini at 1.5-3 bits/dim, otherwise within
  host noise (ship and final share search code yet differ by ±10% single-thread, up to ±30% at 8 threads).
- `output-final-matched-timing/matched-timing.csv`: speed at every matched budget of the final tree (`run.py`,
  seed 0). Versus the previous matches: unchanged within noise, OpenAI 2 bits/dim +11-25%.
