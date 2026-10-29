# Issue #9: is the greedy bit allocator optimal? (Hartmut, 2026-10-29)

Fast-mode recall (R@10, mean over seeds 0-2, 100k base, 1k queries, inner-product truth) against base = c85b0f8.
Code and job scripts: `benchmark/experiments/issue9_empirical/`; eval program: `benchmark/experiments/issue20_additive/ab_eval.cpp`.

- `gap/`: greedy vs exact optimum of the allocator's own objective (predicted distortion / total variance) on the
  seed-0 spectrum (`allocation_gap.cpp`, `job_gap.sh`). Dense greedy equals the exact DP everywhere; every gap is in
  the padded nibble fast layout (2/4 fallback or unused padding).
- `distortion/`: empirical per-rank distortion tables (`DV_TIEREDQ_DISTORTION=raw|shape`) vs the Gaussian table
  (`job_ab.sh`). Recall-neutral (about ±0.35 pt) except GloVe 3.0 bits/dim, a padding effect.
- `oracle/`: exact padded allocator on variance^power (`DV_TIEREDQ_ALLOCATOR=exact`,
  `DV_TIEREDQ_ALLOCATION_POWER`, `job_oracle.sh`). Recall rises with power on SIFT/OpenAI/Gemini (power 1.5: +0.4 to
  +3.0 pt); power < 1 loses 1-4 pt. MSE-optimal allocation is not recall-optimal.

`summary.txt` files are `summarize.py` output; columns per arm are R@10, change in points and stored bits/vector.
- `round2/`: layout-preserving exact allocator (nibble path and padded limit from the unweighted model, so stored
  bits match base) on variance^power 1-2.5 and on query-energy weights (`job_round2.sh`). Power 1.5 peaks
  (SIFT +0.2 to +2.0, OpenAI +0.4 to +1.1, Gemini +0.4 to +1.1 pt at equal stored bits); query-energy weighting
  trails it everywhere. `time-*.tsv` / `timing-summary.txt`: train and add seconds (median of 3 interleaved
  rounds, CPUs 0-7) and batch-32 search QPS (best of 3). Training is unchanged (within ±4%); add grows 12-52% in
  packed cells where steeper allocations widen the top tier to 7-8 bits; search QPS is unchanged or higher.
- `heldout/`: even/odd query split (`job_heldout.sh`, `heldout.py`). Chosen on even queries, power 1.5 also wins
  on odd ones: pooled +0.77 pt even, +0.72 pt odd; per dataset it is chosen on SIFT, OpenAI and Gemini. The
  ScaNN-style parallel weight `1 + η·λ/mean(λ)` trails it (η=1: +0.68 odd; η=0.1: +0.46 odd). The parallel arms were
  rerun with `LANES` after a case-order bug in the first run; they produced no rows then.
