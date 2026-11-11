# Native Qdrant TurboQuant comparison

This focused runner uses Python 3.12/uv for data, calibration and timing, a C++20/nanobind `NOMINSIZE` TieredQ fast binding, and a Rust release `cdylib` loaded through ctypes. It measures compressed-only exhaustive cosine search. It does not require or start a Qdrant server.

The dataset catalogue pins Gemini/DBpedia revision `e14a0f193988e9efb7ccc57b4c615418e6540f62` and both parquet SHA256 checksums. `gemini_data.py` verifies all 100,000 rows, dimensions and finite values, holds out 1,000 seed-0 document vectors, and preserves split IDs and array hashes. These are document queries, not independently embedded search queries.

Clone Qdrant beside the TieredQ checkout (the Cargo path dependency expects this layout), pin `222b2e44ab1c3f17e8246396ead6558e45041df7`, and run `python benchmark/codec_adapters/qdrant_turboquant_plus/patch_qdrant.py ../qdrant` exactly once on that clean dependency checkout. The overlay seeds the otherwise unseedable production calibration sampler and adds allocation/timing observations; it preserves native encoding/scoring and upstream fixed rotation seeds.

From `benchmark/`, on Hartmut, use the existing locked Python environment and a separate native build:

```sh
uv sync --locked --python 3.12
cmake -S codec_adapters/qdrant_turboquant_plus/native -B codec_adapters/qdrant_turboquant_plus/build \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" -DCMAKE_BUILD_TYPE=Release
cmake --build codec_adapters/qdrant_turboquant_plus/build -j8
cd codec_adapters/qdrant_turboquant_plus/native
RUSTFLAGS='-C target-cpu=native -C opt-level=3' \
  CFLAGS='-O3 -march=native -ffp-contract=off' \
  taskset -c 0-7 cargo +1.99.1 build --release --locked -j8
cd ../../..
OPENBLAS_NUM_THREADS=1 taskset -c 0-7 uv run --locked python -c \
  'from pathlib import Path; from data import prepare; from threadpoolctl import threadpool_limits; threadpool_limits(1); prepare(Path("data"), "gemini")'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 taskset -c 0-7 \
  uv run --locked python codec_adapters/qdrant_turboquant_plus/run_experiment.py /absolute/fresh/output
```

Use a fresh output directory for every run. The runner builds the three Qdrant anchors, calibrates integer TieredQ total-code-bit budgets from storage only, and validates all six configurations before the 24-row timing matrix. Calibration retains every measured attempt and selects the largest measured feasible allocation, stopping within 0.5% when possible. The public TieredQ cap remains eight times input dimension. Discontinuities can leave larger gaps.

`verify_data.py DATA_DIRECTORY REPORT_JSON` independently checks prepared hashes, unit norms and full-corpus FP32 argsort truth for ten query rows. `audit_report.py BUNDLE_DIRECTORY` expects the provenance, binaries and truth array archived by the experiment; it audits every raw timing/prediction file and produces three figures and the result README.

Tradeoffs: the Qdrant serving adapter reconstructs its public quantizer from production metadata, retains contiguous compressed codes, and drops build-only storage/metadata/scratch. Its native scorer scores 1,024-vector chunks into a bounded top-10 (the same ranking as a full sort, verified ID-identical), so query-time RAM does not grow with N; TieredQ retains its own native scan/selection scheduling. The shim also supports L2 (`tq_build(..., l2=true, ...)`) for SIFT; the main benchmark uses it through `benchmark/qdrant_codec.py`. This avoids an ANN container or FP32 backing storage while preserving production kernels, but measures these explicit adapters rather than end-to-end database requests. Qdrant parallelism is across queries; TieredQ may also parallelize a single-query scan. See the measured bundle's `INTEGRATION.md` for precision, storage and timing details.
