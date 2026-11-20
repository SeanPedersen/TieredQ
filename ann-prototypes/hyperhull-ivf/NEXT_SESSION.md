# Follow-up prompt (paste into a new Claude Code session in this repo)

Continue the hyperhull-ivf experiment. Read `CLAUDE.md` first (findings + layout), then this file.

**Hard rule:** run ALL compute on host `hartmut` (36 cores, 62 GB, i9-7980XE Skylake-X). The local laptop has
~6 GB free and crashed once. Locally only edit files, then sync and run remotely (remote shell is zsh: split
lists with `${=VAR}`):

```bash
rsync -az --exclude .venv --exclude __pycache__ --exclude .cache --exclude native/build --exclude .git \
  ~/projects/TieredQ/ann-prototypes/hyperhull-ivf hartmut:projects/playground/
ssh hartmut 'cd ~/projects/playground/hyperhull-ivf && uv sync -q && ...'
```

## Status (2026-11-20, end of session 2): router + k-NN graph beats FAISS 1.5–1.6× single-thread, 2.2–2.6× at 32 threads
End-to-end `uv run hyperhull-native-bench` (fresh index, single thread on core 20, 500 test queries), µs/query:
| Recall | Router | Router + 16-NN graph | FAISS IVF1024,PQ192x4fs,Refine(Flat) |
|---|---|---|---|
| ≥0.90 | 68 | **55** | 84 |
| ≥0.95 | 100 | **83** | 135 |
| ≥0.99 | 321 | **234** | 370 |
32 threads q/s: router+graph 110k / 109k / 50k vs FAISS 50k / 43k / 21k. Build 38 s + 1.9 s graph vs 7 s;
extra memory 46 + 19 MB vs 36 MB. Exact graph (FAISS flat, ~5 min build) would give ~208 µs @R≥0.99.

Done this session (correctness: `check_router.py` passed after the kernel/layout changes; graph code has no
Python test yet):
- Projection: K-blocked 6×16 GEMM tile (GCC needed `#pragma GCC unroll` to keep accumulators in registers) 12 → 10 µs.
- Pass 1: 64 B aligned prefix-code rows + per-record aux [bias, o_P, o]; 4-record gather kernel across cell
  boundaries, SIMD score, `U4Weights<NB>` register-resident weights; pass 2 reads one 64 B line. 185 → 160 µs @0.99.
  Pass 1 is now near the single-core streaming floor (~1.7 MB/query) → only fewer/smaller records help.
- Ranking-gap diagnostics + member-level validations (see CLAUDE.md "Fine-cell ranking gap", "Member-level ideas").
- Graph expansion in C++: `SearchParams{hops, expand_from, expand_k, expand_exact}`, `Router::build_graph`
  (self-search, `reverse_edges`), `refine_graph` (NN-descent round), `set_graph`. Bench args
  `T,F,V,R,G,H,P,K,E`, env `HH_GRAPH=k:T,F,V,R`, `HH_GRAPH_REVERSE=1`, `HH_GRAPH_REFINE=fanout,rounds`,
  `HH_GRAPH_REBUILD=...`, `HH_GRAPH_FILE=~/hullexport256/knn32.bin` (exact graph; also graph-recall reference).
  Frontier sweep: `experiments/native/sweep_graph.py <bench> <export> <label> [ENV=..]`.

## Tooling (all on hartmut)
- Standalone bench: `cd native && g++ -std=c++20 -O3 -g -march=native bench/hullbench.cpp cpp/router.cpp -o ~/hullbench`
  `HH_PREFIX_DIMS=128 taskset -c 20 ~/hullbench ~/hullexport256 3 T,F,V,R[,G] ...` → recall, µs/query, stage µs
  (project, coarse, fine, sketch = pass 1 + its selection, refine = pass 2, rerank). `HH_SAME_QUERY=1` = cache-hot
  compute floor (any value, even empty, enables it). `HH_FINE_DIMS=64` ranks fine cells on 64 dims.
- Exported indexes: `~/hullexport256` (m32, PCA-256, 4-bit; reference), `~/hx_m{16,32}_p256_b{2,4}`.
- Python on an export: `uv run python experiments/native/bench_export.py ~/hullexport256 128 T,F,V,R ...`
- Correctness: `uv run python experiments/native/check_router.py` (exhaustive = brute force,
  single and two-pass, threads equal). Selector microbenches: `select_bench.cpp`, `select_parts.cpp` there.
- Profiling: `perf record -F 4999` + `perf report --sort sym|srcline`; no call graphs available.

## What is known (do not redo)
- Single core is limited by ~12 line-fill buffers: scattered ~2 KB reads reach ~8 GB/s, FAISS's long lists ~13.
  Prefetch hint/distance (also per-record in pass 1), gap-merging, streaming top-n: no gain.
- Cell-level hulls cannot approach the oracle in 1536-d (directional spread carries no signal; see CLAUDE.md).
  Only free win at cell level: + β·r² in the fine bias (needs accurate, e.g. PCA-256, fine centroids).
- Graph: self-search graphs miss the hard neighbours; reverse edges fix most of it. Rebuild-with-expansion and
  extra NN-descent rounds are not worth their build time.

## Next steps (in order)
Done: graph in Python bindings (`HullRouter.build_graph`, `search(..., hops, expand_from, expand_k, expand_exact)`),
parallel reverse-edge merge, graph checks in `check_router.py` (all pass), end-to-end bench above.
1. **Build time** (38 s vs 7 s; now the weakest point): time two-level k-means, PCA SVD, projection, C++ layout.
2. **Cheaper expansion**: exact distances dominate (E per hop × ~0.6 µs); try sketch-only first hop + one exact
   round, expand_exact adaptive (stop when sketch score > current kth bound).
3. **Residual 2-bit PCA-256 sketch** (validated: 64 B, V@0.99 80 vs 200 for current pass 1) → likely drops
   pass 2 and helps expansion scoring. Needs `native_arrays` change + re-export.
4. **Fine ranking**: β·r² in fine bias + two-stage fine ranking (128 → 256 dims on survivors), validated −30% vectors.
5. Memory: graph k=8 or delta-coded ids; sketch aux as fp16.

## Later
- Build time (25 s vs 7 s): time the steps (two-level k-means, PCA SVD, projection, C++ layout) first.
- Memory (46 MB vs 36 MB): sketch bias/residual as fp16, or PCA-192 sketch.
- Repeat on GloVe / another dataset and a non-Skylake-X CPU (AVX-512 licence effects differ).
- Nothing is committed yet (git repo has no commits) — ask before committing.
