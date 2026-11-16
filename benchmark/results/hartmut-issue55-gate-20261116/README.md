# #55 non-regression gate: aae87c7 (A) against fba7ef5 (B)

Remote root: `/home/sean/experiments/TieredQ-issue55-gate-20261116` (hartmut, AVX-512), 2026-11-16, under the shared
benchmark lock. B is the step-1 library change: fixed-capacity fast tiles and the public `TieredQ::map_scores`. Both
trees are installed as Python packages (`.venv-A`, `.venv-B`). `study.py` runs, in one window:

1. B's C++ tests (ctest) and Python suite on AVX-512. The first attempt failed only because the shared test
   workspace kept a stale `examples/` directory (`results/tests-failed-1/`); the rerun synced it and passed.
2. `kernel_bench.cpp`: single-thread scans of the shared fast-scan kernels over one 1M-row `TieredQFastBlocks`
   block (synthetic Gaussian rows; 768-D at 1.75 bits/dim and 1024-D at 1 bit/dim, both nibble lookups). Order: A, B,
   B, A on CPU 2.
3. Per dataset, `gate.py`: flat builds in ABBA order (VIBE arxiv: 1.75 bits/dim, 2,048 lists; Qwen/MS MARCO 8.84M:
   1 bit/dim, automatic 64K lists with the router, from the SQ8 store), nested builds A then B (library defaults,
   8-bit tier on NVMe), then A's saved flat and nested indexes served by fresh processes in ABBA order. Single queries
   run alone on CPU 2 (best of 3 passes; arxiv k = 100, Qwen k = 10), then one 32-thread batch.

`REPORT.md` (from `summarize.py`) has every number. The npz files with ids and scores stayed on hartmut, and the
indexes were removed.

## Results

- **Results unchanged.** Every build of either tree wrote byte-identical files (sha256: flat ABBA builds, nested A and
  B index and tier), and every serve returned the ids and scores of serve 1A. B loads A's saved indexes.
- **Kernels.** Whole-run block scans: B / A −1.6% (768-D) and −0.8% (1024-D), with A-A spread 2.0% and 1.1%. That is
  no slowdown. On B, the same rows as 256-document tiles scan 4-10% faster than one 1M-row block, since the scratch
  scores stay in cache.
- **Single-query QPS, flat.** arxiv −0.4 to +1.0% and Qwen −1.4 to +1.7% per nprobe, inside the A-A spread of each
  point (0.2-1.1%) or within 1 point of it.
- **Single-query QPS, nested.** arxiv −0.1 to −1.5%. Qwen −1.3 to −3.1%, where the A-A spread is 2.4-4.9%. The Qwen
  deficit comes only from the second pair. 2B matches 1A at every nprobe (111.3 against 111.0 QPS at nprobe 1024),
  while 3B (106.5) and 4A (113.8) diverge as host load fell from 8 to 1.6 during the window. The nested search does not
  run the one changed search function (`finish_scores`). This is host drift, not attributable to the change.
- **Batch QPS (32 threads).** −0.6% / +0.1% (arxiv flat / nested) and −0.7% / −0.8% (Qwen), against A-A spreads of
  0.0-4.8%.
- **Build and memory.** Flat builds take 21.8-22.3 s (arxiv) and 204.8-220.0 s (Qwen, where A's first build was the
  slowest). Nested builds and tier writes are equal within 2.6 s. `storage_bytes` and serving RssAnon are equal to
  0.001 GiB. Qwen's peak build RSS is 17.6 GiB for 1A and 19.4 GiB for the other three, which follows build order,
  not tree.

The gate passes for the step-1 change. Later commits change only `TieredQFastBlocks` (partly filled blocks,
`reset`), which neither `TieredQIndex` nor `TieredQIVFIndex` uses.
