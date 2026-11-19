# #45: production assignment bound (`assignment_bound_dimensions`) in IVF builds (hartmut, 2026-11-19)

Source: this commit (`src/tieredq_ivf_bound.h`, `RestrictedAssigner`). The final restricted assignment scores each
row's first probed super in full, the other probes' members by an exact lower bound on the centroids' top 224
principal directions plus residual norms, and rescores survivors on contiguous binary16 rows. `bench.sh`: the
automatic training (bound on from `bounded_lists` = 65,536) vs the same training with `assignment_bound_dimensions` 0.

## TieredQNestedIVFIndex, Qwen/MS MARCO 8.84M x 1024, 64K lists (#52 defaults), order off s0, on s0, on s7, off s7

| | off s0 | **on s0** | off s7 | **on s7** |
|---|---:|---:|---:|---:|
| Build (tier write excluded) | 176.3 s | **159.6** | 175.1 | **159.7** |
| Train / assign / encode | 62.3 / 53.7 / 49.6 | 62.4 / **36.9** / 49.6 | 62.2 / 51.7 / 50.4 | 62.7 / **36.1** / 50.4 |
| Peak build RSS | 15.348 GB | 15.418 | 15.385 | 15.450 |
| R@10 nprobe 96 / 256 / 1024 | 0.9472 / 0.9699 / 0.9836 | identical | 0.9448 / 0.9675 / 0.9852 | identical |

Recall@10 is identical at every nprobe (96-1024) on both seeds; p50 within host noise. Builds -9%, assignment -30%,
peak RSS +0.07 GB (binary16 member rows for rescoring and the bound's panels).

`results-gather/`: the first production attempt rescored survivors by gathering panel columns (d / 2 cache lines per
candidate instead of d / 32): assignment 58.4 vs 53.5 s, slower than no bound. Replaced by contiguous rows.

## TieredQIVFIndex, MS MARCO 1M x 1024 (2 bits/dim, router off, 8 threads), `results/ivf1m.jsonl`

| lists | assign, bound off (s0 / s1) | assign, bound 224 (s0 / s1) | recall@10 |
|---:|---:|---:|---|
| 8,192 | 4.6 / 4.6 s | 7.6 / 7.7 s | identical at every nprobe |
| 32,768 | 8.8 / 8.7 s | 12.1 / 11.6 s | identical at every nprobe |

With ~512 (8K) or ~2K (32K) candidates per row the projection and the bound cost more than they save, and the
basis fit is not amortized over 1M rows, so the recipes bound only from 65,536 lists (`bounded_lists`). The first
auto-nobound run at 8K (train 108 s) is a cold-start outlier; its seed-1 twin trained in 6.2 s.
