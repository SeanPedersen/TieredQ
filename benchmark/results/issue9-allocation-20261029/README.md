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
