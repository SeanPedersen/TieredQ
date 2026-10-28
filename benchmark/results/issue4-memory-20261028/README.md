# Issue #4: disk vs RAM bits/vector, and the streamed TieredQ loader

This answers the issue #4 comment asking for separate **bits/vector on disk** and **bits/vector in RAM (max usage)**. While measuring, we found that TieredQ's index load briefly needed about 2× the index in memory. The streamed loader fixes that. This bundle reports the matrix both before and after the fix.

Hartmut (Intel i9-7980XE, AVX-512), CPUs 0–7, GCC 16.2.1, `-O3 -march=native -ffp-contract=off`. Library base `f73d394`. "After" adds the streamed loader in `src/tieredq_io.h`, `src/tieredq_fast_storage.h` and `src/tieredq_index_io.cpp`. RaBitQ is the unchanged official flat adapter; its new `load` only reads its existing saved format. Same four corpora, splits and ground truth as the main issue #4 run. Seed 0 and 1,000 queries.

## Method

`benchmark/measure_memory.py` builds each cell in one fresh process. A second fresh process reloads the saved index and runs the full timed workload: 3 warmups and 3 repeats at threads 1/8 × batches 1/32. All columns are bits per stored vector (bytes × 8 / N).

- **Disk**: saved index file, plus the SIFT TieredQ L2 lift.
- **RAM**: steady RSS growth from just before the reload to just after it.
- **Peak RAM**: the larger of the reload peak and the query-workload peak, both measured as VmHWM growth from a reset (`/proc/self/clear_refs`). It includes query scratch, OpenMP threads and allocator overhead.
- **Build peak**: VmHWM growth from the loaded FP32 corpus through training, encoding and packing.

The Python runtime, queries and ground truth are excluded. Fixed per-process costs are divided by N like every other shared overhead. Recall is the 8-thread batch-32 seed-0 value. It reproduces the main issue #4 seed-0 rows, so the reloaded indexes are the benchmarked ones.

## Result (after the loader fix)

All columns are bits/vector. Full per-cell CSV: [after/memory.csv](after/memory.csv). The table including accounted RAM, load and query peaks, and build peak is in [after/memory-table.md](after/memory-table.md).

| Dataset | Bits/dim | TieredQ disk | TieredQ RAM | TieredQ peak RAM | TieredQ R@10 | RaBitQ disk | RaBitQ RAM | RaBitQ peak RAM | RaBitQ R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sift | 1 | 218 | 225 | 267 | 0.5006 | 224 | 226 | 244 | 0.2429 |
| sift | 2 | 331 | 336 | 417 | 0.7622 | 416 | 418 | 436 | 0.5493 |
| sift | 4 | 587 | 591 | 675 | 0.9333 | 672 | 674 | 692 | 0.8493 |
| glove | 1 | 310 | 341 | 565 | 0.4121 | 352 | 369 | 534 | 0.4049 |
| glove | 2 | 510 | 541 | 776 | 0.6719 | 672 | 689 | 854 | 0.6559 |
| glove | 4 | 902 | 933 | 1,178 | 0.9011 | 1,184 | 1,201 | 1,366 | 0.8821 |
| openai | 1 | 2,298 | 2,569 | 2,831 | 0.8939 | 1,633 | 1,649 | 1,861 | 0.8126 |
| openai | 2 | 3,948 | 3,975 | 4,789 | 0.9474 | 3,233 | 3,249 | 3,461 | 0.9041 |
| openai | 4 | 7,097 | 7,129 | 7,978 | 0.9839 | 6,305 | 6,321 | 6,533 | 0.9664 |
| gemini | 1 | 1,110 | 1,256 | 1,496 | 0.8337 | 864 | 876 | 1,075 | 0.7192 |
| gemini | 2 | 1,826 | 1,852 | 2,626 | 0.9252 | 1,696 | 1,708 | 1,907 | 0.8559 |
| gemini | 4 | 3,399 | 3,427 | 4,218 | 0.9784 | 3,232 | 3,245 | 3,444 | 0.9508 |

Equal-memory readout. This compares measured points only, with no interpolation. A configuration "wins" if it has higher recall at the same or lower memory.

- **SIFT and GloVe:** TieredQ wins on disk, RAM and peak RAM at 2 and 4 bits. At 1 bit, TieredQ also wins on SIFT. On GloVe 1-bit, its peak RAM is 6% above RaBitQ's for +0.7 points of recall.
- **OpenAI and Gemini:** TieredQ has higher recall at each nominal budget but uses more memory. At RaBitQ's measured disk, RAM or peak-RAM budgets, the TieredQ point that fits has slightly lower recall. For example, OpenAI TieredQ 1-bit at 2,831 peak bits gets 0.894, against 0.904 for RaBitQ 2-bit at 3,461. Gemini TieredQ 2-bit at 2,626 gets 0.925, against 0.951 for RaBitQ 4-bit at 3,444.
- **Recall@10 ≥ 0.95 by peak RAM:** RaBitQ reaches it with OpenAI 4-bit (6,533) and Gemini 4-bit (3,444). TieredQ reaches it with OpenAI 4-bit (7,978) and Gemini 4-bit (4,218). Neither codec reaches it on SIFT or GloVe within this grid.

## Streamed loader: before vs after

`TieredQIndex::load` used to read the whole file into a buffer, copy the model and payload out of it, then parse the payload into the fast-scan storage. At its peak the process held two full copies of the index. It now reads the 80-byte header and the checksummed model, then streams the payload directly into its final vectors through a 1 MiB buffer, checksumming on the fly. Corruption is still reported as a checksum mismatch, and full-mode loads stream row by row. Saved files and the in-memory layout are unchanged.

[ab/table.md](ab/table.md): the same saved index files served by the old and new library in fresh processes, alternating old/new/old/new.

- Top-10 IDs and accounted bytes are identical in all 12 TieredQ cells.
- The load peak roughly halves. For example, OpenAI 1-bit goes from 4,768 to 2,617 bits/vector, and SIFT 4-bit from 1,167 to 593. The overall TieredQ peak RAM, now set by the query workload, drops by 26–44% on SIFT/OpenAI/Gemini and 1–34% on GloVe. GloVe 1-bit was already query-bound. See [before/memory.csv](before/memory.csv) vs [after/memory.csv](after/memory.csv).
- QPS and latency are unchanged. Search code is identical, and transparent-huge-page backing after load is identical. The first A/B showed after/before QPS ratios of 0.82–1.19× (mean 0.984), with the extremes on noisy 8-thread single-query cells. A tighter single-thread re-check on the 1-bit cells, with 6 alternations per side, gave median ratios of 0.976–1.019×, within the 1–4% run-to-run spread of either side ([timing-probe/](timing-probe/)).

## Remaining TieredQ memory above its index

- **Query scratch, about 3–10 MB per process.** The code-lookup layouts (OpenAI and Gemini at 2 and 4 bits) treat a batch of 32 queries as one tile shared across 8 document-segment workers. Each worker needs its own score scratch of 32 × 4,096 floats. Reducing it would mean retuning the document tile size, a throughput trade-off, so it is left unchanged. The cost does not grow with N: it is about 800 bits/vector at 100K vectors and under 100 at 1M. RaBitQ's equivalent is about 2.5 MB.
- **Derived lookup state.** TieredQ's steady RAM exceeds its disk size by about 30 bits/vector on GloVe and 145–270 on the 1-bit OpenAI/Gemini cells, from projection panels and codebooks rebuilt at load.
- **Build peak.** TieredQ needs roughly 10,600–82,400 bits/vector to build at these sizes (about 1 GB for OpenAI at 100K). This is mostly a fixed training working set. RaBitQ's build peak is close to its index size. The build peak for SIFT TieredQ includes the lifted FP32 copy made by the L2 adapter.

## Files

- `after/`, `after-run/`: final matrix with the streamed loader. `before/`, `before-run/`: the same matrix with the original loader and split load/query peaks. `before-combined-peak/`: the first original-loader run, before load and query peaks were separated.
- `ab/`, `ab-run/`, `ab_streamed_load.py`, `summarize_ab.py`: same-file old/new loader A/B.
- `timing-probe/`: huge-page and interleaved single-thread timing probes.
- Each `cell-*` folder holds `build.json` and `serve.json`, including per-schedule recall, diagnostic QPS and a top-10 ID hash. Memory runs are not headline timing runs.

Limitations: seed 0 only for memory (storage is fixed-stride per configuration), RSS includes allocator behaviour of glibc on this host, and Hartmut is shared, so timing ratios have a few percent of noise.
