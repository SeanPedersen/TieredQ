# Packed IVF lists (#53): identical results, 8.5% smaller at 64K lists, 1-3% slower there

Remote root: `/home/sean/experiments/TieredQ-issue53-packedlists-20261111`. A is 86f003d plus the heap fix
(`../hartmut-heapretention-20261111`), with lists padded to 32 documents. B adds packed lists: a scan starts each list
on the 32-document boundary at or below its first row and gives top-k only the list's own rows. B saves payload
version 6, and older files keep loading. `ab_index.py` builds a flat IVF once per variant and saves it. Partitions and
codes are deterministic, so they are identical in A and B.
- VIBE arxiv: 1.75 bits/dim, 2,048 lists.
- Qwen/MS MARCO 8.84M: 1 bit/dim (the nested default's prefix), automatic 64K lists with the router, from the SQ8 store.

Fresh processes then serve each index in ABBA order. Single queries run alone on pinned CPU 2 with one thread, and
batches as one call on 32 threads; best of 3, at k = 10 and 100. `compare.py` writes `REPORT.md`, with every serve's
QPS at its end. The serves ran twice. First with the embeddinggemma worker running (about 130% CPU,
`results-loaded-host/`), where batch QPS was noisy (A's two serves differ by up to 16%). Then on the same saved indexes in the window
of `../hartmut-quiethost-20261111` (`results/`), with the production containers pinned off CPUs 2/20 and 19/35. The
worker still ran on the other cores. The tables below lead with that run.

Before the benchmark, CTest (2/2) and pytest (220) passed on the packed source. They include
`padded_fixture_test`: `tests/fixtures/tieredq_ivf_padded_*`, written by `make_padded_fixture.py` with 86f003d (v4 and
v5 payloads, one with a router). The loaded files return their recorded ids; rebuilt in place, packed, they return
the same ids and scores, save as version 6 and round-trip. The full-probe tests (packed IVF = flat index bit for bit,
1/2/4/6 bits, every metric, range checkpoint) also pass. The first test run failed on a storage comparison in the new
test (`results/tests-failed-1`): a rebuilt index also holds its router's encoder state. The test now compares two
loaded indexes.

## Results

| | arxiv, 2K lists | Qwen 8.84M, 64K lists |
|---|---|---|
| Rows per list, mean | 657 | 135 |
| Padding in A (share of rows) | 2.37% | 11.65% |
| File, A → B | 238.0 → 232.7 MiB (−2.2%) | 1,553.8 → 1,421.2 MiB (**−8.5%**) |
| Ids and scores, every serve, both k | identical | identical |
| Single-query QPS, B against A | −2.4% … +0.2% (mean −1.0%) | −0.5% … −3.2% (mean −1.5%); **−0.8 to −1.6% from nprobe 256** |
| Batch QPS, B against A | −12% … +3.9% (A spread up to 10.6%) | −6.9% … +2.5% (mean −3.0%); −0.7 to −4.7% at nprobe 256 and 1024 (A spread up to 2.7%) |

Loaded host (`results-loaded-host/`, the first run):

| | arxiv, 2K lists | Qwen 8.84M, 64K lists |
|---|---|---|
| Single-query QPS, B against A | −1.6% … +1.2% (mean −0.5%; the 2nd serve, a B, ran low) | 0 … −2.6%; **−2.1 to −2.6% from nprobe 256** (A's serves within 0.3% at nprobe 1024) |
| Batch QPS, B against A | within the noise (−14% … +1.4%, A spread up to 16%) | +3.1% … −3.6%; −2.5 to −3.6% at nprobe 256 and 1024 (A spread under 1%) |

`storage_bytes` falls by less than the file (−0.3% / −6.6%) because B's also counts the codec (about 5 MB, see the
heap study) and A's does not.

## Why the 64K-list case is slower

The kernels score whole 64-document chunks from a 32-aligned start. In A a list's padding (15.4 rows on average) sat
inside that rounding, so it cost storage but almost no scan time. In B the rows between the aligned start and the
list's first row are scored and dropped: 15.5 on average, 11.5% of all rows at 64K lists and 2.4% at 2K. That is the
same count as the padding saved, now as scan work. It costs 1-2.5% per query at high nprobe (isolated cores / loaded host) and about 3% in batches, against 8.5% less
index (12% fewer stored codes). In QPS per GB that is about +5% at 64K lists, and +2% at 2K, where speed is unchanged.
A smaller start alignment would shrink the head (16 documents: 7.5 rows). The AVX2 lookup kernels process 32-document
chunks, so that needs kernel work.

**Acceptance (#53):** identical ids and scores, smaller storage and files, and old files load: met (both runs). QPS
within noise: met at 2K lists. At 64K lists not met, and the rerun on isolated cores confirms the cost: −0.8 to −1.6% single from
nprobe 256 (−2.1 to −2.6% loaded), about −3% in batches (−2.5 to −3.6% loaded). Accepted as a QPS-per-GB win.
