# Issue #14: lookup grouping and serving-aware allocation

This isolated experiment keeps production sources, public formats, Gaussian levels and the normal allocator unchanged. `prepare.py` modifies a **fresh source copy**, never the working checkout. Stack: C++20/CMake/OpenMP/nanobind, Python 3.12 through the existing uv environment, NumPy and threadpoolctl. No Brandt coder is implemented.

Variants:

- `TIEREDQ_EXPERIMENT_ALLOCATION=marginal`: unchanged Gaussian marginal-distortion allocation.
- `...=log`: allocate the next bit to the largest `log2(sigma) - allocated_bits`, breaking ties by PCA rank. Sigma is derived from the pre-rotation residual covariance, with the same eight-bit cap, total budget, centroids, tier rotations, and Gaussian levels. This is only Brandt's allocation rule inside TieredQ.
- `...=cap4`: use the marginal rule with a four-bit coordinate cap when feasible. This measures the narrow-representation tradeoff. Above `4*D` this policy falls back to marginal; the driver excludes such redundant cases.
- `TIEREDQ_EXPERIMENT_GROUPING=byte`: combine the two original quantized nibble lookups into one 256-entry int32 table, indexed by the byte already stored. Integer sums, original scale boundaries, accumulation order and metadata are unchanged. Keep the original int8 tables in query scratch and explicitly count the additional table memory.
- `wide.py` in a second fresh source copy replaces packed-coordinate register lookup with a 256-entry repeated FP32 codebook and gather. The same immediate-shift/period unpacking and coordinate accumulation order remain. Wide coordinates remain whole-coordinate groups; smaller tiers on this path deliberately retain one-coordinate accumulation to avoid a floating-point reassociation confound. No arbitrary cross-coordinate bit slices are introduced.

The byte experiment trades register lookup for memory gather on AVX-512. This is one concrete byte-table implementation, not an optimality claim about every possible byte layout. It does not benchmark a new AVX2/NEON backend.

## Reproduce

From a **new isolated copy** containing `src`, `include`, `cmake`, `CMakeLists.txt`, `benchmark/experiments`, and `benchmark/l2_transform.py`:

```sh
python3 benchmark/experiments/serving_allocation/prepare.py "$PWD"
cmake -S benchmark/experiments/tiny_budgets -B build \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=/path/to/benchmark/.venv/bin/python
cmake --build build -j8
```

Use `run.py --mode lookup`, then `--mode allocation`, then `--mode timing`, each with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 taskset -c 0-7 \
  /path/to/benchmark/.venv/bin/python benchmark/experiments/serving_allocation/run.py \
  --data /path/to/existing/benchmark/data --native build \
  --previous /path/to/tiny-budget/results --output results --mode lookup
```

The driver requires an existing tiny-budget bundle with its frozen adapter, recipes and seed-0 indexes. It checks that retrained default recipes match those previous recipes byte for byte. It refuses neither old result directories nor duplicate rows: **always use a new output directory and run each mode once**.

`validate.py` checks append/load, document-block tails, zero-query ties, all-document scores/ranks, prior top-10 IDs, exact budgets and fixed Gaussian tables. `reconstruction.py` measures 2,048 base vectors disjoint from each seed's codec training rows. `costs.py` fits a small candidate selector on those base reconstructions and 32 base-vector query proxies, never held-out queries or truth. Its constraint grid is actual bytes/vector and measured single-query latency; `analyze.py` chooses candidates from those costs, then evaluation quality remains a separate report.

`cross_dataset.py --mode quality` runs seed-0 GloVe/OpenAI checks at four/eight bits per dimension, 8,192 training rows. Its `--mode timing --variant register` precedes `--variant byte` using the second build. `wide_sift.py` similarly tests mixed/wide 256/512/1032-bit SIFT cases. Their `--native`, `--data`, `--results`, and (SIFT) `--previous` arguments select existing artifacts.

For the wide build, copy the prepared source root to another fresh directory, run `wide.py /path/to/that/copy`, then configure/build its tiny-budget extension in a new build directory. For diagnostics, copy the prepared primary root to another fresh directory, run `profile.py /path/to/diagnostic/copy`, and build with `-DEXPERIMENT_PROFILE=ON`. Run `diagnose.py` with that diagnostic build. Table time is nested inside preparation time; phase counters must never supply primary latency evidence.

`audit.py` checks effective compiler flags and derives packing/padding/group/table footprints. `analyze.py results` summarizes the JSONL files. The result bundle contains the exact remote `finish-study.sh`, frozen source trees, compiler audits, manifests, machine metadata, raw measurements and final report.
