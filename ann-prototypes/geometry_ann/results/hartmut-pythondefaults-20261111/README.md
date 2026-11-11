# End-to-end low-RAM build and serve through the Python API (#52)

Remote root: `/home/sean/experiments/TieredQ-issue52-pythondefaults-20261111`. TieredQ 9b158c6 plus the #52 working
tree, built as the Python package (Release, liburing). Qwen/MS MARCO 8.84M × 1024 from the memory-mapped SQ8 store
(`base.i8`, `scales.f32`, `offsets.f32`), never as float32. `build.py` runs the build on 32 threads. `serve.py` serves
in a separate process on pinned CPU 2 with one OpenMP thread, one query at a time, over the 750 held-out queries. Its
anonymous RSS is the serving RAM. The tier sits on the NVMe SSD (direct reads, 32 in flight).

- `defaults`: `TieredQNestedIVFIndex.build(codes=, scales=, offsets=)` with no other option, then
  `write_tier(path, codes=, ...)`, `save()`, and `load()` / `attach_tier(path, reads="direct")` /
  `search(q, k=10, nprobe=...)`. That is a 1-bit prefix, 64K lists, fast list training, the router, an 8-bit fine
  suffix tier and 40 candidates.
- `twobit`: the same with `prefix_bits=2048` and `candidates=20`, the earlier winner.

`results/` is after the list-sample RNG fix (`../hartmut-bisect-20261111/`). `results-before-fix/` is the first run.
`results/defaults-contended/` is a serve that overlapped other load on the host (p99 up to 208 ms); its recall equals
the re-serve's.

## Results

| | defaults (1-bit, 40 cand.) | twobit (2-bit, 20 cand.) |
|---|---|---|
| Build (index + tier) | 302 s (208 + 94) | 379 s (248 + 131) |
| Peak build RSS | 21.0 GB | 20.9 GB |
| Serving RAM (anonymous RSS) | 1.72 GB | 2.98 GB |
| Disk (index + tier) | 1.63 + 7.93 GB | 2.89 + 6.76 GB |

Recall@10 @ p50 (ms):

| nprobe | defaults | C++ 1-bit (hartmut-defaults, seed 42) | twobit | C++ 2-bit (hartmut-ssdtier, seed 42) |
|---:|---|---|---|---|
| 96 | 0.9381 @ 2.36 | 0.9437 @ 2.19 | 0.9392 @ 2.43 | 0.9436 @ 2.36 |
| 128 | 0.9480 @ 2.60 | 0.9513 @ 2.38 | 0.9492 @ 2.80 | 0.9513 @ 2.67 |
| 192 | 0.9560 @ 3.10 | 0.9627 @ 2.89 | 0.9573 @ 3.51 | 0.9628 @ 3.35 |
| 256 | 0.9620 @ 3.83 | 0.9684 @ 3.37 | 0.9633 @ 4.26 | 0.9685 @ 4.04 |
| 512 | 0.9741 @ 5.71 | 0.9773 @ 5.14 | 0.9757 @ 7.09 | 0.9776 @ 6.63 |
| 1024 | 0.9804 @ 9.09 | 0.9843 @ 8.36 | 0.9823 @ 12.4 | 0.9843 @ 11.6 |
| 2048 | 0.9872 @ 13.8 | 0.9876 @ 13.1 | 0.9891 @ 20.6 | |
| 4096 | 0.9895 @ 24.3 | 0.9912 @ 22.1 | 0.9915 @ 36.8 | |

## Acceptance

- Build, about 400 s or less on 32 threads: met (302 s and 379 s).
- Serving RAM, about 3.4 GB: met. The 2-bit build serves in 2.98 GB, the 1-bit default in 1.72 GB.
- p50 within about 5%: the 2-bit build is 3-7% slower than its C++ reference. The 1-bit default is 7-14% slower, in
  a window with load on the host (load average 17-24, p99 spikes). Not cleanly met.
- Recall@10 within 0.002 at matched nprobe: not met against the single reference partition. The 2-bit build is
  0.002-0.005 lower, the 1-bit default 0.003-0.007. The bisect shows why the criterion is too tight: the partition seed
  alone moves recall by up to 0.0044 at 8.84M, and the references (both seed 42) sit at the top of that spread. The
  explicit C++ build at seed 7 gives 0.9496 / 0.9647 / 0.9816 at nprobe 128 / 256 / 1024, where these Python builds land.

The 1-bit default against the 2-bit winner, both through Python: 42% less serving RAM and 1.25× faster build, at
0.001-0.002 lower recall at matched nprobe and 7-34% lower p50 from nprobe 128.
