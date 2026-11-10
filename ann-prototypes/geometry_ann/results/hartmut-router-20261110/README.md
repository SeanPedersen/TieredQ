# TieredQ IVF built from scratch at Faiss's training budget, with the two-level router (#47, #48, #51)

Remote root: `/home/sean/experiments/TieredQ-issue44-router-20261110`. `tieredq_ivf_qwen.cpp` builds a 2-bit
`TieredQIVFIndex` (inner-product codec, 4-bit scan cap) on Qwen/MS MARCO 8.84M × 1024 from the SQ8 store. Lists are
trained in the library: 4 Lloyd iterations on 40 sample rows per list, as Faiss trains in our harness, with restricted
assignment (256 supers, 16 probes) in training and build. The two-level router comes next (lists/128 supers, 2-bit codes,
exact re-rank of the first 256 lists), and the built index is saved to `indexes/`. Measurement: top-40 SQ8 re-rank,
750 held-out queries alone on one pinned core, plus a 32-thread batch pass. The given-partition models (Faiss's Lloyd
lists) were dropped as unneeded: the scratch partition reached the same recall.

| Lists | Codec | List training | Assignment | Encoding | Router | **Total** | Faiss IVF-HNSW / flat |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64K | 0.9 s | 82 s | 53 s | 89 s | 6 s | **232 s** | 1,621 / 1,634 s |
| 128K | 0.9 s | 254 s | 93 s | 98 s | 12 s | **457 s** | 3,052 / 4,729 s |

Peak build RSS was 20.9 GB at 64K and 32.4 GB at 128K. Index sizes: 2.73 GB plus a 0.18 GB router at 64K, and
3.15 GB plus 0.33 GB at 128K.

With exact routing, the scratch partition matches the Faiss Lloyd partition (`hartmut-tieredqivf-20261110`): 0.952 vs
0.950 at nprobe 96, 0.970 vs 0.971 at 256, 0.984 vs 0.983 at 768. The router at expansions 16/32 lost 0.013–0.025
recall at nprobe 128–256; `hartmut-routertune-20261110` raised the expansion. Raw curves are in `results/*/run.csv`, and
batch rows are in `run-batch.csv`.
