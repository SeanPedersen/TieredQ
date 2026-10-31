# Issue #22: allocation power by budget (2026-10-31)

Allocation power α ∈ {1, 1.25, 1.5} at 0.25 / 0.5 / 0.75 / 1 bits per dimension (prefix code budgets d/4 ... d),
256-list IVF candidate recall (true top 10 among k' at nprobe 32 / 128, k' 20 / 100), OpenAI-1M (1,536-d, document
queries) and MS MARCO-1M (1,024-d, real and document queries). α is chosen on even query ids and reported on odd
ones (`QUERY_PARITY`), as in #9. Job: [`job_alpha_budget.sh`](../../experiments/issue22_query_alloc/job_alpha_budget.sh);
reference-LAPACK eval binary, untimed, Hartmut CPUs 8-31.

| Bits/dim | Best α on even ids (summed over sets and metrics) | Held-out gain vs 1.5, odd ids: mean (range) |
|---|---|---|
| 0.25 | 1 | +1.0 pt (+0.4 to +1.7) |
| 0.5 | 1 | +0.7 pt (-0.0 to +1.8; +1.3 to +1.8 on real questions) |
| 0.75 | 1.25 (1 ties) | +0.1 pt (-0.1 to +0.4) |
| 1 | 1.5 | 0 |

The library default for inner-product and cosine models is now α = 1 + clamp(bits/dim - 0.5, 0, 0.5): 1 at and
below 0.5 bits per dimension, 1.25 at 0.75, 1.5 from 1 bit per dimension up (unchanged there, as #9 and the #4 1M
check chose). Flatter weights code more PCA ranks, which pays off when the budget cannot cover the spectrum.
