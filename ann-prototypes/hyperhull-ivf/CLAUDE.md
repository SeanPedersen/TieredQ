# hyperhull-ivf

Experiment: can "hyper-hull" summaries (outer bounds / shape sketches) make IVF search cheaper
than plain centroid routing at equal Recall@10? Python 3.12 + uv, numpy; faiss-cpu + scipy (dev).

## Run
- `uv run hyperhull-ivf [glove100k|glove|dbpedia|synthetic]` – IVF heuristic routers, cell hulls, VA hulls (cost model)
- `uv run hyperhull-bench [method ...]` – vs FAISS on GloVe-100k: recall vs cost & wall time, build s, index MB, build peak MB.
  Each method runs in its own subprocess (clean peak RSS). Results → `.cache/bench_results.json`.
- Fast iteration lab: `experiments/lab/lab.py [estimator-prefix ...]` (dataset caches are written next to it)
  – 8 neighbouring GloVe k-means clusters (~2.9k vectors), cells = clusters or ~33-vector blocks,
  estimators scored vs oracle (true min) on held-out queries; params tuned on a train split.

- `uv run hyperhull-native-bench [n_coarse] [m] [n_queries]` – C++ hull router (± k-NN graph expansion) vs FAISS IVF-PQ4fs+refine, wall time.

## Native kernel (`native/`, uv workspace member `hyperhull-native`)
C++20 + nanobind, built by scikit-build-core on `uv sync` (`-O3 -march=native`, 256-bit AVX2+FMA kernels;
512-bit only with `-DHH_AVX512` – on Skylake-X they drop the core to the lowest turbo licence and lose).
`cpp/kernels.hpp` SIMD kernels (4-bit×int8 `maddubs` rows, projection GEMM micro-kernel) · `cpp/select.hpp`
top-n selection (sampled threshold + SIMD compress + bisection on int keys; `select_about` approximate) ·
`cpp/router.hpp/.cpp` search: per 32-query block projection + coarse scores as small GEMMs → coarse top T →
4-bit fine hull top ≈F → pass 1 streams a prefix sketch (first prefix_dims) of the selected fine cells →
pass 2 completes the 256-dim sketch for top `refine` → exact rerank of top V.
Optional k-NN graph (`build_graph` self-search + reverse edges, `refine_graph`, `set_graph`): after the rerank,
`hops` rounds sketch-score the graph neighbours of the best `expand_from` and score the best `expand_exact` exactly.
Sketch records: 64 B aligned prefix / suffix code rows + per-record aux [bias, o_P, o]; pass 1 = 4-record gather kernel. Layout: coarse cells / fine
cells renumbered along nearest-neighbour chains of their centroids (cells picked together are contiguous).
`cpp/bindings.cpp` `make_router(native_arrays(index, pca_dims=256), prefix_dims=128)`; `search(..., refine=R)`.
`native/bench/hullbench.cpp` standalone bench on an exported index (`T,F,V,R` args, env HH_PREFIX_DIMS).
Rebuild after C++ edits: `uv sync` (cache-keys watch cpp/**). Exclude `native/build` when rsyncing.

Data: `~/data/ann` (glove_base.npy 1.18M×100, dbp_base.npy 305k×1536 f16, query .f32 files).
GloVe-100k subset + ground truth cached in `.cache/glove100k.npz`.

## Layout (`src/hyperhull_ivf/`)
- `__init__.py` experiment runner · `bench.py` FAISS benchmark + error-ball method
- `data.py` synthetic data · `real_data.py` GloVe/DBpedia loaders
- `ivf.py` k-means + contiguous lists (chunked, memmap-friendly)
- `routers.py`/`summaries.py` heuristic list ranking · `evaluate.py` cached top-K, curves, cost interpolation
- `hulls.py` sound cell hulls (PCA box × ring × sphere; shared-subspace block hulls)
- `two_stage.py` two-level k-means index (coarse-major, fine-minor) · `combined_router.py` cost-model sweep of the
  combined hull router · `faiss_full.py` FAISS PQ4fs baseline (cost model) · `native_router.py` export to C++
  arrays (4-bit packed codes) · `native_bench.py` wall-time C++ router vs FAISS
- `bounded_search.py` two-level cell-hull search · `va_search.py` per-vector SQ-box hulls · `tightness.py` bound diagnostics

## Experiments (`experiments/`, run with `uv run python experiments/<dir>/<script>.py` from this folder)
- `native/` C++ router tooling: `check_router.py` (correctness incl. graph), `export_index.py` / `bench_export.py`
  (standalone-bench exports), `rank_gap.py` / `rank_tune.py` (fine-cell ranking gap to the hull oracle),
  `graph_gap.py` / `sketch_gap.py` (member-level idea validation), `sweep_graph.py` (hullbench frontier sweeps),
  kernel / selector microbenches (`*.cpp`).
- `lab/lab.py` fast estimator lab · `diag/` early bound / margin diagnostics.

## Cost model
total = routing (L) + codes scanned × code_bits/32·D + exact float distances. Wall time is single-thread;
Python reference loops (va*, errball re-rank) are not speed-comparable with FAISS C++.

## Findings (2026-11-19, GloVe-100 100k unless noted)
- Heuristic list re-ranking (hull points, PCA box, sphere) ≈ centroid ordering; routing overhead makes it worse.
- Cell-level hulls do not work in high intrinsic dim: a prunable cell's nearest member is only ~1.15× the
  10th-NN distance (synthetic: 1.66×), so a sound cell bound would need ≥87% tightness; ours reach 0.4–9%.
  Approximate cell hulls (Gaussian/ellipsoid/union-of-balls/global-PCA box, vertices, point-min up to R=32)
  close ≤20% of the oracle gap. Flat GloVe spectrum → NN decision lives outside top PCA dims.
- Per-vector hulls work: SQ4 boxes (sound, exact at eps=0) reach R≥0.99 at ~15% of IVF-Flat cost (model),
  but FAISS IVF-SQ4+refine is equivalent and its non-SIMD scan is slower in wall time than IVF-Flat.
- FAISS IVF-PQ4 fast-scan + refine is the real bar (R≥0.99 at ~0.18 ms/query).
- Error-ball hull on IVF-PQ4 (store ||x - x̂|| per vector): sound ball too loose (re-ranks ~95% of pool);
  probabilistic ball (τ σ, RaBitQ-style) at τ=4 keeps recall (0.999 vs 1.0) with 2.4× fewer re-ranks.

## Findings – DBpedia OpenAI-1536 100k (`hyperhull-bench --dataset dbpedia100k`, lab with HULL_LAB_DATA=dbpedia)
- FAISS IVF-PQ4fs+refine dominates (R≥0.99: ~107 exact + 25.6k 4-bit codes). Our per-vector hulls lose here
  (VA4: 365 exact + 19.3k codes; error-ball re-ranks its whole 1000 pool) – bounds too loose vs tiny NN gaps.
- All IVF methods must touch ~17k vectors/codes for R≥0.99 → list selection is the shared bottleneck.
- Cell-level hulls *do* have headroom on DBpedia (anisotropic text embeddings), unlike GloVe:
  8 close clusters, prune@R≥0.99 (oracle-kth threshold): oracle 0.81, centroid 0.17,
  shrunk sphere (||q-c|| - 0.4·rms)^2 0.60 (cost 1/cell), PCA-16 point-min approx 0.69.
  At ~34-vector blocks centroid is already 0.74 (oracle 0.93); best PCA-16 point sketch 0.76.
  Sound cell bounds (box×ring, sphere) still prune ≈0.
- Progressive validation (`hyperhull-prune dbpedia100k [n_lists]`, running kth, params tuned on 100 train
  queries, scored on 100 test queries) – the lab gains do NOT survive:
  256 lists, R≥0.99 total: fixed IVF 16.6k, adaptive IVF 16.5k, shrunk sphere (tuned α=0 → same), PCA-16 sketch 21.0k.
  1024 lists: fixed IVF 8.8k (finer lists help IVF itself), adaptive 11.9k, shrunk sphere α=0.6 16.4k
  (train looked better: overfit), PCA-16 sketch 25.5k. Lab "prune@R" used the true kth, which flatters
  calibrated estimators; with a running kth, centroid ordering is as good or better.

## Hull routing: clusters scanned to reach recall (`hyperhull-route`, `hyperhull-cluster`)
Metric: scan clusters in hull-score order; **vectors scanned** (fair across unequal cell sizes) and
total = vectors + routing distances, for Recall@10 (1000 DBpedia queries,
params tuned on 500, scored on 500; 256 cells). Heavy runs go to host `hartmut` (36 cores, 62 GB):
rsync the project (exclude .venv/.cache), data in ~/data/ann, `.cache/dbpedia100k.npz` already there.
- k-means 256: oracle 4.0 / 6.0 clusters @R≥0.95/0.99, centroid 13.5 / 59.5.
- **Sub-ball hull** (hull = cluster's own finer k-means, m balls; est = min distance to a sub-centroid, alpha=0):
  m=8: 12.0 / 44.2, m=32: 10.8 / 36.2 (−39% clusters @0.99). Routing cost L·m (checked: shortlist variant).
- Cheap sub-ball routing: shortlist (sub-ball check only on the T nearest centroids) + sub-centroid distances
  in an r-dim PCA space of the sub-centroids. Total (vectors + routing) @R≥0.95 / 0.99, centroid = 6182 / 25670:
  T=128,m=32 full-dim 9248 / 20766; pca128 T=128 5854 (−5%) / 20331 (−21%); pca256 T=128 6044 / 18868 (−26%).
  m=64/128 scan fewer vectors but routing outweighs it. Centroid ordering still wins @R≥0.90.
- **Full scale, two-stage k-means** (`hyperhull-twostage 1024 500` on hartmut; 305k, 1024 coarse × m fine):
  scan *fine cells* ranked by fine-centroid distance (PCA-r space) inside the T nearest coarse cells.
  Total (vectors + routing) @R≥0.90 / 0.95 / 0.99; flat IVF-1024 = 4138 / 8764 / 32922:
  1024×32 pca128 T=64: 2494 / 4567 / 23374;  pca256 T=128: 2949 / 4380 / 15133;  pca256 T=256: 3626 / 5020 / 13234.
  (−40% / −50% / −60%). Ranking whole coarse cells by sub-balls (B) gains little; the win is fine-cell scanning.
  Headroom: oracle-fine needs only 113 / 129 / 150 vectors; we scan ~1000 / 2400 / 10600 → routing now
  dominates at R≤0.95 and fine-centroid order is ~18× off the oracle at R≥0.95.
- **Combined hull router** (`hyperhull-combined 1024 32 1000` on hartmut, ~2.5 min): query projected once
  to global PCA-256; coarse centroids (r dims) → T coarse shortlist → fine centroids → top F fine cells →
  per-vector s-dim sketch (+ residual-norm term, γ) → exact scan in sketch order. Config tuned on 500 train,
  reported on 500 test queries. Total (vectors + routing), flat IVF-1024 → combined:
  R≥0.90: 4233 → 506 (59 vec + 447 routing; T=32 F=128 s=128 γ=0.5)
  R≥0.95: 9140 → 873 (204 + 669; T=64 F=256 s=128)
  R≥0.99: 32650 → 3384 (351 + 3033; T=256 F=2048 s=128)   ≈ −88…−90%. Vectors scanned ≈ fine-cell oracle.
  v2 (quantized sketch, per-dim uniform SQ, exact residual norm): 4-bit sketch → 386 / 540 / 1547
  (R≥0.90 / 0.95 / 0.99; 69 / 87 / 384 vectors scanned). 8-bit: 395 / 661 / 1783.
  Ball-shaped hulls (shrink by cell rms, coarse β, fine α) tuned to 0 → no gain over centroids.
  v3 (fine centroids 4-bit SQ in PCA space, coarse in 64 dims): **268 / 363 / 952** (53 / 102 / 430 vectors +
  215 / 261 / 522 routing). Smaller sketches (32/64 dims), two-pass sketch (32→128 on top P) and adaptive
  stopping (est ≤ c·est_K) were never selected → no gain. Routing floor now: query projection (128 dims = 128)
  + coarse 43 + fine ~85 + sketch reads ~200 at R≥0.99. Oracle (vectors only, zero routing) 129 / 157 / 214.
- **Wall time, end-to-end, router + 16-NN graph vs FAISS** (2026-11-20 session 2, `hyperhull-native-bench`, fresh
  index, both single-thread on core 20, 500 test queries): µs/query router / router+graph / FAISS IVF1024,PQ192x4fs,Refine(Flat)
  R≥0.90 68 / **55** / 84 (1.52×) · R≥0.95 100 / **83** / 135 (1.62×) · R≥0.99 321 / **234** / 370 (1.58×).
  32 threads q/s: router+graph 110k / 109k / 50k vs FAISS 50k / 43k / 21k (2.2–2.6×).
  Build: router 38 s + graph 1.9 s (32 thr) vs FAISS 7 s; extra memory 46 + 19 MB vs 36 MB.
  Graph configs: R≥0.90 T=16 F=50 V=20 R=150 H=1 P=10 E=5 · R≥0.95 T=32 F=100 V=30 R=200 H=2 P=20 E=10 ·
  R≥0.99 T=256 F=1000 V=60 R=400 H=1 P=30 E=20 (expand_k=16).
- **Wall time, C++ router vs FAISS IVF1024,PQ192x4fs,Refine(Flat)** (session 1, before graph / kernel work; `hyperhull-native-bench`,
  hartmut, single thread pinned to one core, both swept on the same 500 test queries): µs/query
  R≥0.90 74 vs 85 (1.14×), R≥0.95 102 vs 136 (1.33×), R≥0.99 350 vs 367 (1.05×); 32 threads ~1.5× FAISS q/s.
  Router config @0.99: T=256 F=1792 V=120 R=800, PCA-256 sketch, prefix 128. Build 25 s vs 7 s; extra memory
  46 MB vs 36 MB. FAISS IVF-Flat needs 2.3 / 3.3 / 12.7 ms at the same recalls.
  What mattered (Skylake-X, single core ≈ 12 line-fill buffers ⇒ ~8 GB/s on scattered ~2 KB reads vs ~13 GB/s
  for FAISS's long lists): two-pass prefix sketch (−30% sketch stage), fast selection (was ~25% of query time:
  64-bit div, nth_element), bitmap instead of sorts, projection/coarse as GEMM (20→12 µs), AVX2 not AVX-512,
  chain-ordered layout (−4…9%). No effect / worse: prefetch hints & distance, merging nearby cells into longer
  reads (gap), streaming top-n selector, 2-bit sketch, m=16, 64-dim prefix, 64-dim fine ranking (≈ neutral).
- **FAISS IVF1024,PQ192x4fs,Refine(Flat), full 305k** (`hyperhull-faiss-full`, same test queries/cost model):
  total 1144 / 1262 / 1964 (exact 40 / 80 / 320). Combined 4-bit router: −66% / −57% / −21%.
  Caveat: cost model only (no wall time); FAISS fast-scan is SIMD-optimized, our router is a Python model.
- **Fine-cell ranking gap to the hull oracle** (2026-11-20, `experiments/native/rank_gap.py
  ~/hullexport256` + `rank_tune.py`, full 305k, 1024×32; train = queries 0–499, test = export queries; ranking
  only, every member of a scanned cell counts). Vectors scanned for R≥0.90 / 0.95 / 0.99 (test):
  router (4-bit PCA-128 fine, T=256) 1189 / 3015 / 16065 · same, no shortlist 1189 / 2988 / 12732 ·
  4-bit PCA-256 cell means 990 / 2504 / 10440 · exact full-dim centroid 1024 / 2318 / 10338 · **oracle 119 / 135 / 151**.
  + β·r² (cell rms², free: folds into fine bias), β tuned on train: full-dim centroid 705 / 1879 / 7559;
  4-bit PCA-256 T=256 889 / 2433 / 11225 (T=1024: 877 / 2383 / 9699); 4-bit PCA-128 (current) only 1084 / 2904 / 14880.
  Directional spread (iso d·rms, anisotropic σ_c(u)·d from per-cell PCA-256 variance, ×√(2 ln n)) tunes to α=0:
  σ_c(u)/rms is 0.055 for GT cells vs 0.056 for peers — members' offsets are ~orthogonal to q−c in 1536-d, so no
  cell summary sees which members lean toward the query. GT-cell centroid rank pct 50/90/99 = 3 / 63 / 682:
  the R≥0.99 tail is per-member, only a member-level scan finds it. Cell hulls are capped at ~50× the oracle here.
- **Member-level ideas validated** (2026-11-20, ranking-only, test queries, `experiments/native/`):
  `graph_gap.py` (exact 32-NN graph, `~/hullexport256/knn32.npy`): scan top F0 cells (router order), take best P
  found (exact), add their k-NN (1–2 hops). Recall / vectors touched vs plain router 1189 / 3015 / 16065 @0.90/0.95/0.99:
  F0=25 P=10 k=16 1 hop 0.906 / 428; F0=25 P=30 k=32 2 hops 0.965 / 757; F0=200 P=30 k=32 2 hops 0.987 / 2862;
  F0=400 P=30 k=16 2 hops 0.990 / 5294 (≈3–4× fewer vectors; top-P by exact distance is optimistic).
  `sketch_gap.py` (pool = top 1265 router cells, V to keep 0.99 of pool GT, best γ): abs4 p128 64 B (current pass 1)
  200; abs2 p128 32 B 350; res4 p128 64 B 163; res2 p128 32 B 261; **res2 p256 64 B 80**; abs4 p256 128 B 45.
  Residual = SQ of z − ĉ (4-bit PCA-256 cell mean), per-dim clipped uniform (2-bit levels ±0.5σ, ±1.5σ).
- **Graph expansion in C++, wall time** (2026-11-20, `hullbench` T,F,V,R,G,H,P,K,E + `sweep_graph.py`, core 20,
  500 test queries, best of ~1800 configs). µs/query @R≥0.90 / 0.95 / 0.97 / 0.99:
  plain 68 / 106 / 190 / 299 · exact 32-NN graph (FAISS flat, ~5 min build) 52 / 75 / 111 / 206 ·
  self-built 16-NN graph (`build_graph` = router self-search 64,128,60,400, 1.8 s on 32 threads, recall 0.91 vs
  exact) 57 / 90 / 132 / 291 → graph quality decides R≥0.99. Self-search graph recall vs build time (32 thr):
  16,32,30,200 0.79 / 0.85 s · 32,64,40,300 0.86 / 1.1 s · 64,128,60,400 0.91 / 1.8 s · 128,256,100,800 0.95 / 3.4 s.
  Expansion cost ≈ exact distances of E per hop (~0.6 µs each); E=40 vs 20 no recall gain.
  A self-search graph misses exactly the "hard" neighbours the expansion needs (correlated errors). Fixes
  (graph recall / build on 32 thr / µs @R≥0.99 in a fine T=256 sweep; plain 299, exact graph 208):
  plain 0.912 / 1.8 s / 279 · NN-descent round fanout 8 (`refine_graph`) +0.02 recall per 0.8 s ·
  rebuild with expansion (2nd self-search using the graph) 0.93–0.94 / 3.2–4.3 s, no better ·
  **reverse edges** (`build_graph(..., reverse_edges=true)`: r found c ⇒ r offered to c) 0.938 / 2.1 s / **230**;
  + refine 0.958 / 3.1 s / 225; search 128,256,100,800 + reverse + refine 0.978 / 4.6 s / 222.
  **Frontier with the 2.1 s reverse-edge graph** (wide sweep): 53 / 81 / 117 / 230 µs @R≥0.90 / 0.95 / 0.97 / 0.99
  vs plain 69 / ~100 / 189 / 299 (FAISS IVF-PQ4fs+refine earlier: 85 / 136 / – / 367). Graph memory +20 MB (k=16 u32).
- Gaussian hull (local PCA-8, κ=1.25): 13.0 / 48.3. Shrunk sphere = centroid (α→0). Local PCA box,
  global PCA-16 sketch: worse than centroid.
- Hull-native partitions lose to k-means: balanced PCA hyperplane tree (centroid 34.3 / 89.3, exact polytope
  bound 76 / 175 — huge polytopes in 1536-d), greedy k-center (unbalanced 1–3107; sub-ball m=32 19.2 / 53.8).
