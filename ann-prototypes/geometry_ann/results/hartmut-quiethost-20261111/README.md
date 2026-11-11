# Python against C++ on one saved 8.84M index, isolated cores (#52 item 4 and 5c)

Remote root: `/home/sean/experiments/TieredQ-issue52-quiethost-20261111`. Source 6298356 (heap fixes and packed lists),
from `../hartmut-packedlists-20261111`'s package build. Qwen/MS MARCO 8.84M × 1024.
- `build.py` builds the 1-bit nested default through the Python API with defaults only: SQ8 memmap in, 32 threads,
  tier on NVMe.
- The same saved index is then served in ABBA order: `serve.py` (Python), `serve_nested` (C++, `serve_nested.cpp`),
  C++, Python. Each serve runs alone on pinned CPU 2 with one thread, the tier read directly with 32 reads in flight,
  750 held-out queries, 40 candidates.
- `batch.py` runs all queries in one call on 32 threads, best of 3.
- Each step's `host.json` records the load average and IO pressure at its start. The index and tier were deleted
  afterwards; `results/build/files.json` keeps their sizes.

Host: the two production containers were pinned to CPUs 3-18 and 21-34, which keeps CPU 2, its sibling 20, and 19/35
free. They were not paused. The embeddinggemma worker ran at about 100-130% CPU on the other cores, so the shared L3,
memory bandwidth and SSD were not exclusive. The load average was 3-5 at the builds and the later serves. The first
serve started with load 19 and IO pressure 12% (avg10), right after the tier write and save.

The first attempt failed at the driver build (`results-failed/driver-parse-error`). `std::vector<T> values(size_t(width))`
declared a function (the most vexing parse). It now reads `static_cast<size_t>(width)`.

## Results

p50 (ms) per serve in ABBA order. Recall@10 is the same in all four serves, and also the same as
`../hartmut-pythondefaults-20261111`'s Python build at every nprobe: the build is deterministic, so this is the same
partition.

| nprobe | recall@10 | Py 1 | C++ 2 | C++ 3 | Py 4 | Py / C++ | Py against pythondefaults | C++ against the C++ reference |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.9381 | 2.35 | 2.26 | 2.44 | 2.35 | +0.2% | −0.2% | +7.3% |
| 128 | 0.9480 | 2.63 | 2.59 | 2.60 | 2.59 | +0.6% | +0.4% | +9.1% |
| 192 | 0.9560 | 3.18 | 3.04 | 3.10 | 3.11 | +2.3% | +1.3% | +6.2% |
| 256 | 0.9620 | 3.68 | 3.70 | 3.64 | 3.63 | −0.4% | −4.6% | +8.9% |
| 512 | 0.9741 | 5.60 | 5.74 | 5.39 | 5.49 | −0.4% | −2.9% | +8.3% |
| 1024 | 0.9804 | 9.40 | 9.66 | 8.76 | 8.95 | −0.4% | +0.9% | +10.1% |
| 2048 | 0.9872 | 14.65 | 14.41 | 13.88 | 14.04 | +1.4% | +3.9% | +8.0% |
| 4096 | 0.9895 | 25.16 | 24.38 | 23.82 | 23.77 | +1.5% | +0.7% | +9.0% |

The "Py / C++" column compares the mean of the two Python serves with the mean of the two C++ serves. The C++
reference is `../hartmut-defaults-20261111` onebit-64k: seed 42, explicit C++ calls at 9b158c6, padded lists.

**The 7-14% gap was not the binding.** On the same index, Python and C++ are within −0.4% to +2.3%.
The same server's two serves differ by up to 10% (C++ at nprobe 1024), so the binding cost is below what this setup
can resolve. The Python p50 also lands within −4.6% to +3.9% of the earlier serve that ran under load 17-24. Served from C++, this
index is 6-10% slower than the seed-42 C++ reference, at 0.003-0.007 lower recall. Packed lists account for about 1.5%
of that (`../hartmut-packedlists-20261111`). The rest is not resolved. The reference ran hours earlier with its host
load unrecorded, and its index is deleted. Host state alone moved this window's serves by up to 10% (the first serves,
at load 19 after the tier write, against the later ones at load 3-5), and the worker still ran beside them. So the
rest may be host state as much as a difference between the builds (partition seed, list training, source version).

| | this run (6298356) | `../hartmut-pythondefaults-20261111` (9b158c6) |
|---|---:|---:|
| Build, index + tier | 336 s (218 + 117), save 10 s; ABBA mean 312 s for both versions | 302 s (208 + 94) |
| Peak build RSS | 20.98 GB | 21.0 GB |
| Index file / tier file | 1.49 / 7.93 GB | 1.63 / 7.93 GB |
| Serving RAM (anonymous RSS, Python, 1 thread) | **1.58 GB** | 1.72 GB |

The serving RAM falls by 8.1% (heap fixes and packed lists), and the index file by 8.5%. The C++ driver's anonymous
RSS (3.06 GB) includes the whole file, which the driver keeps as a byte vector. Without it, the driver's RSS is the
same as Python's.

This build took 11% longer, most of it in `write_tier` (117 s against 94 s). An ABBA of both versions shows that this
is host state, not code (`../hartmut-tierwrite-20261111`): build plus tier write averages 312.6 s at 9b158c6 and
312.0 s at 6298356. 9b158c6's own `write_tier` runs took 94 s and 106 s.

### Batch QPS of the SSD configuration (32 threads, first measurement)

| nprobe | 96 | 128 | 192 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| recall@10 | 0.9381 | 0.9480 | 0.9560 | 0.9620 | 0.9741 | 0.9804 | 0.9872 | 0.9895 |
| QPS | 3,746 | 3,396 | 2,903 | 2,567 | 1,725 | 1,068 | 642 | 360 |
| Gain over one thread | 9.0× | 9.0× | 9.2× | 9.6× | 9.7× | 9.8× | 9.2× | 8.6× |

The anonymous RSS is 1.67-1.68 GB on 32 threads. The gain over one thread uses the single-query mean latency (serve 4).
The 32 threads run on 16 physical cores with their hyperthread siblings. At nprobe 96 that is about 150K tier reads
per second.

## Open

- C++ latency gap to the seed-42 reference: only an ABBA serve of both indexes in one window, with the services
  paused, settles it. Not planned, since it changes no code. Until then, quote this run's numbers for the Python
  default.
