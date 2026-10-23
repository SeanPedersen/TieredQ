# Exact TieredQ, fast TieredQ and RaBitQ on GloVe200

Same data on hartmut (Intel i9-7980XE): 1,183,514 normalized GloVe200 vectors, the same first 500 held-out queries, k=10 and supplied exact ground truth. Search is measured at one core and 18 threads only. Exact and fast TieredQ use the identical existing 800-bit model: 200 four-bit coordinates and 256 centroids. RaBitQ uses four total bits per coordinate with 256 lists and full probe.

| Method | 1-core QPS | 18-thread QPS | Recall@10 |
|---|---:|---:|---:|
| Exact TieredQ | 50.9 | 891.6 | 0.9002 |
| Fast TieredQ | 91.8 | 1394.3 | 0.8992 |
| RaBitQ 4-bit | 120.9 | 1574.9 | 0.8728 |

Fast TieredQ is 1.80x faster on one core and 1.56x faster at 18 threads than exact TieredQ. Recall changes from 90.02% to 89.92%: a 0.10 percentage-point drop. Fast/exact top-10 set overlap is 97.82%. Net ground-truth hits change by five across 5,000 returned neighbors; individual neighbor substitutions are more numerous, as the overlap shows.

Exact here means the normal FP32 scorer of TieredQ’s compressed representation (`search` / `search_batch`). It does not mean brute-force scoring of the original uncompressed float vectors, so its Recall@10 need not be 100%. Fast uses the opt-in quantized fast cache (`search_fast` / `search_batch_fast`). No model is retrained between the two TieredQ modes.

RaBitQ retains its one-bit screening plus conditional extra-bit refinement, but probes all 256 lists: IVF does not reduce the vectors visited. Full probing still includes native IVF list/routing overhead. The comparison is not iso-recall.

## Measurement protocol

- Median aggregate QPS across three independent processes per method/setting. Each process warms up eight queries and times three complete passes over 500 queries.
- One-core runs: CPU 2, one query per call. Eighteen-thread runs: CPUs 0–35, batches of 500 with OMP_PLACES=cores, OMP_PROC_BIND=close and OMP_DYNAMIC=FALSE. OPENBLAS_NUM_THREADS=1 and MKL_NUM_THREADS=1.
- Timings include query preparation, scan, top-k selection and search scratch. Loading, cache construction, training and evaluation are excluded. Exact TieredQ does not build a fast cache.
- RaBitQ and fast TieredQ were run sequentially with alternating engine order. Exact TieredQ was added afterward using the same data, library, affinity, warmup and repetition settings. The three engines were not interleaved together. Background services remain running.
- Both models use the existing 20,000-row training file, with no query rows used for training. Persisted RaBitQ metadata confirms 200 input dimensions, 256 padded dimensions, 256 lists, four total bits and inner-product metric. TieredQ codes 200 coordinates, so nominal bit width does not imply equal byte footprints.

## Validation

Every RaBitQ query reports 1,183,514 visited rows. Within each method, IDs and score payloads are byte-identical across all repeats and between one core and 18 threads. All results have ten finite, ordered scores and valid IDs. RaBitQ query rows are byte-identical to the original first 500 rows; ground-truth IDs are widened from uint32 to int64 with exact value equality verified.

## Reproduction and artifacts

Artifacts are in ~/experiments/TieredQ-linear-fast-20261023/rabitq-reference on hartmut: scripts/, build/, data/, indexes/, results/ and logs/. The reference reuses unchanged native zvec/RaBitQ objects from ~/experiments/issue6-parallel-rabitq-20261023; isolated drivers adapt dimensions and warmup. TieredQ drivers link the identical optimized library from the bounded-memory experiment. GCC 16.2.1, Release, OpenMP and AVX-512 on this host. No production code changed for these reference measurements.

Run the fast/RaBitQ comparison with ../optimized/.venv/bin/python scripts/run.py. Compile the exact driver with ../optimized/.venv/bin/python scripts/prepare.py, then run ../optimized/.venv/bin/python scripts/run_exact.py. Finally run ../optimized/.venv/bin/python scripts/report.py. The index builder refuses to overwrite an existing index.

Raw/summary evidence: [rabitq-glove200-reference-results.json](benchmarks/rabitq-glove200-reference-results.json). One dataset, one model per codec and one shared host; no confidence claim about other datasets or training seeds follows.


## Budget fairness

The comparison matched four nominal bits per original coordinate, not an effective byte/vector budget. TieredQ stores 200 coded coordinates; RaBitQ pads 200 inputs to 256 coordinates before storing four bits each.

| Budget component | TieredQ | RaBitQ |
|---|---:|---:|
| Quantization code bytes/vector, before metadata | 100 | 128 |
| Complete persisted index bytes/vector | 105.31 | 160.56 |

On-disk totals include shared models and container overhead. RaBitQ also stores eight-byte keys and a four-byte mapping per vector; standalone TieredQ uses implicit row IDs. These identification costs must be normalized or reported separately in a codec-only comparison.

The RAM comparison differs: fast TieredQ retains 105.00 bytes/vector of original planes plus 109.02 bytes/vector of cache arrays, totaling 214.02 bytes/vector before shared models, query projection, scratch and allocator overhead. RaBitQ's base/extra arrays, keys and mapping total 160.16 bytes/vector before shared models and other overhead. These are retained representation sizes, not a measurement of process RSS.

Therefore the speed/recall table establishes same-data performance at nominal four-bit settings. It does not establish equal-storage or equal-RAM efficiency. A budget comparison should measure stored and retained-search bytes independently, normalize ID requirements, select configurations by measured bytes/vector, and compare recall/QPS along those budget frontiers.
