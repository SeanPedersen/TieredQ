# arxiv follow-up: recall@10, nested memory and candidates, THP (#43, #41, #42, #27)

Remote root: `/home/sean/experiments/TieredQ-issue41-arxivfollowup-20261111`. Same harness and host setup as
`../hartmut-arxivdefaults-20261111` (TieredQ b0148dd, VIBE arxiv 1.34M × 768, pinned CPU 2, 32 build threads, THP
off unless `host`). `vibe_arxiv.py` adds `K`, the per-mapping private memory that grew across `fit()` (smaps), nested
USS per step plus serialized size, and `WARMUP_ROWS`, which builds a small index first so one-time process state
precedes the baseline. Every curve is in `REPORT.md`.

## Recall@10: the 4K-list loss is not about k

| lists | recall@10 @ QPS (selected nprobes) |
|---|---|
| 1,024 | 0.9132 @ 2,148 (12), 0.9401 @ 1,314 (24), 0.9493 @ 865 (40) |
| 2,048 | 0.9020 @ 2,485 (12), 0.9296 @ 1,743 (24), 0.9402 @ 1,458 (32), 0.9479 @ 1,104 (48) |
| 4,096 (old automatic) | 0.9003 @ 1,961 (16), 0.9301 @ 1,535 (32), 0.9393 @ 1,263 (48), 0.9513 @ 826 (96) |

2K is 14-20% faster than 4K at 0.93-0.94, as at k = 100. Together with `../hartmut-qwen1mlists-20261111` this set the
new `automatic_lists` rule (2K lists at 1-1.34M).

## Nested (1-bit prefix, 8-bit tier on SSD, direct reads)

- **k = 10, default 40 candidates:** 0.9341 @ 807 QPS (np16), 0.9686 @ 748 (32), 0.9852 @ 649 (64), 0.9938 @ 528
  (128), 0.9983 @ 385 (256). Flat at 1.75 bits stops at 0.955, so above it nested is the only TieredQ option, at
  about half of flat's QPS.
- **k = 100, 150 candidates:** 0.9750 @ 359 (np64), 0.9855 @ 320 (96), 0.9903 @ 302 (128), 0.9943 @ 277 (192). 200
  candidates give 0.9800 @ 291 at np64. 150 loses about 0.005 at matched nprobe and is about even at equal recall, so
  the library default becomes `max(40, 2k)` = 200 at k = 100. Study A showed 200 within 0.001 of 400.
- **Memory:** USS +193 MiB after `build` (`storage_bytes` 144 MiB; the serialized index is 159 MB, so the refinement
  is only about 8 MB). After `write_tier` it reaches +274 MiB: about 30 glibc thread arenas of 2.3 MiB each from the
  parallel tier encoding. `write_tier` now releases freed heap (`malloc_trim`) as `build` does. That makes the
  earlier 0.269 GiB about 0.19 GiB. Not yet re-measured.

## #42 at its own configuration (2 bits/dim, 2,048 lists)

USS 0.303 GiB with THP off against 0.305 GiB under the host's THP `always` (+2 MiB; the issue measured about
+60 MiB). Per-mapping growth matches except one region (+1.4 MiB). It no longer reproduces.

## Flat USS above `storage_bytes`

Cold builds sit about 45 MiB above `storage_bytes`. The warm build (`flat175-2048-warm`) measures 0.251 GiB against
0.272 cold, so about 21 MiB is one-time process state: one 20.7 MiB anonymous region, OpenBLAS 3.8 MiB and the module
1.7 MiB. VIBE charges it to whichever index a process builds first. About 23 MiB per index remains in `[heap]` after
the build's trim, about 9% of storage. Unattributed.
