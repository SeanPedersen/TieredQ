# Issue #22 task 3.1: query energy on the data PCA, and the allocation-power check (2026-10-31)

Go/no-go diagnostic for query-aware bit allocation on MS MARCO-1M (Cohere embed-english-v3.0, 1,024-d; data in
[issue22-msmarco1m-20261031](../issue22-msmarco1m-20261031/README.md)). **Verdict: no-go.** Real questions carry only
slightly more energy on weakly coded ranks than doc queries, and the allocation lever helps both query sets alike.

Scripts: [`experiments/issue22_query_alloc/`](../../experiments/issue22_query_alloc/) (`query_energy.py`,
`job_diagnostic.sh`, `job_alpha_probe.sh`). Hartmut, CPUs 8-31, untimed.

| Path | Contents |
|---|---|
| `energy.md` | `query_energy.py` output: energy shares per rank band and tier, predicted error per allocation rule |
| `nesting.txt` | `ivf_eval tieredq-nesting 512,768,1024`: tier layouts of the eval's models (32,768-row seed-0 sample, 256 centroids) |
| `msmarco-train-queries-20000-v1.json` | manifest of the train-question sample (`msmarco1m_data.load_train_queries`, seed 42, disjoint from dev) |
| `alpha-probe-{real,doc}/` | `ivf_eval` with `ALLOCATION_POWER` 1.25-2.5, prefixes 512 / 768, 256 lists, fp32 rerank |

## Query energy (`energy.md`)
mu_r = E[(q.u_r)^2] on the residual PCA ranks u_r of the eval's training sample; shares of the total per query set.

| 512 prefix (300 coded ranks) | data lambda | real | doc | train |
|---|---|---|---|---|
| 4 + 3 bit (ranks 0-51) | 0.29 | 0.29 | 0.35 | 0.30 |
| 2 + 1 bit (52-299) | 0.50 | 0.42 | 0.40 | 0.42 |
| uncoded (300-1023) | 0.21 | 0.29 | 0.25 | 0.28 |

- Real questions put 3.9 pt more energy than doc queries on the uncoded ranks (and 1.3 pt more on the 1-bit tier).
  The larger difference is at the top: doc queries (held-out passages) concentrate on the 4/3-bit ranks, where
  TieredQ is strongest, while real questions follow the data spectrum there.
- Train questions match the real dev questions within about 1 pt in every band: a usable fitting sample.
- Both query sets are much flatter than the data spectrum on the tail (mu/lambda 2-7x on ranks 512+).
- Under the Lloyd-Max error model sum mu_r lambda_r D(b_r), real questions have ~28% more relative score error than doc
  queries at every budget; re-allocating at equal bits with weights lambda * mu_train is predicted to cut it 5-8% on
  *both* sets. RaBitQ's isotropic 1-bit error (approximate) is 0.120 real vs 0.101 doc: TieredQ 512's error is ~1% below
  RaBitQ's on real and ~5% below on doc, matching the larger real-query loss of the 512 prefix in #22.

## Allocation power on real and doc queries (`alpha-probe-*`)
Candidate recall (true top 10 among k'), delta vs the default 1.5 in points:

| Prefix | α | real n32 k′20 | doc n32 k′20 | real n32 k′100 | doc n32 k′100 | real n128 k′20 | doc n128 k′20 | real n128 k′100 | doc n128 k′100 |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 1.25 | 0.848 (+0.8) | 0.929 (+0.3) | 0.926 (+0.3) | 0.978 (+0.1) | 0.885 (+0.8) | 0.943 (+0.5) | 0.978 (+0.5) | 0.995 (+0.1) |
| 512 | 1.5 | 0.840 | 0.926 | 0.922 | 0.978 | 0.877 | 0.938 | 0.973 | 0.994 |
| 512 | 2 | 0.827 (-1.3) | 0.913 (-1.3) | 0.916 (-0.6) | 0.975 (-0.3) | 0.861 (-1.6) | 0.925 (-1.3) | 0.965 (-0.7) | 0.991 (-0.3) |
| 512 | 2.5 | 0.810 (-3.0) | 0.898 (-2.8) | 0.906 (-1.6) | 0.971 (-0.7) | 0.842 (-3.5) | 0.909 (-2.9) | 0.952 (-2.0) | 0.987 (-0.7) |
| 768 | 1.25 | 0.901 (+0.1) | 0.962 (+0.0) | 0.934 (+0.1) | 0.980 (+0.0) | 0.948 (+0.4) | 0.978 (+0.0) | 0.989 (+0.1) | 0.998 (+0.0) |
| 768 | 1.5 | 0.899 | 0.962 | 0.933 | 0.980 | 0.945 | 0.978 | 0.988 | 0.998 |
| 768 | 2 | 0.891 (-0.8) | 0.957 (-0.5) | 0.931 (-0.2) | 0.980 (-0.0) | 0.934 (-1.1) | 0.971 (-0.7) | 0.985 (-0.3) | 0.997 (-0.0) |
| 768 | 2.5 | 0.881 (-1.8) | 0.949 (-1.3) | 0.929 (-0.4) | 0.979 (-0.1) | 0.923 (-2.2) | 0.964 (-1.4) | 0.982 (-0.6) | 0.996 (-0.2) |

- The error model's direction is wrong: the lambda * mu weighting it favours is steeper than lambda^1.5 and codes
  fewer ranks (251 vs 300 at 512), while steeper allocation loses recall on both sets. Mean squared score error is
  not what top-k recall needs, as #9 already found for alpha = 2.
- A flatter allocation (α = 1.25) helps the 512 prefix on both sets: +0.8 pt real, +0.3 to +0.5 pt doc at k' 20.
  The real-specific part is ~0.3-0.5 pt of the 6.6 pt real-query loss against RaBitQ; at 768 every α in 1.25-1.5 ties.

## Conclusion
- **No-go for query-aware allocation (#22 lever).** The real-query loss of sub-1-bit prefixes is not mainly
  misallocated bits: doc queries are unusually favourable to a data-variance codec (energy on its best-coded ranks),
  and RaBitQ's isotropic error does not care where a query's energy sits. Remaining levers are more prefix bits
  (768 ties RaBitQ on both sets) or a larger k'.
- **Side finding:** α = 1.25 beats 1.5 at sub-1-bit budgets on both query sets (+0.3 to +0.8 pt), as at 1 bit/dim on
  OpenAI-1M (#4 α check, +0.23 pt). A budget-dependent default power is a small separate lever.
