#!/usr/bin/env bash
# #33 grid on hartmut (~2.5 min): build the bridge, then measure Gemini-768 99K and OpenAI-1536 100K at 1/2/4/8 bits/dim.
# Usage: job.sh WORKSPACE DATA   (WORKSPACE holds src/ = this repo; DATA holds {gemini,openai}/base.f32)
set -euo pipefail
workspace=$1 data=$2
cd "$workspace"
uv venv -q -p 3.12 venv && uv pip install -q -p venv/bin/python numpy nanobind
cmake -S src/benchmark/experiments/issue33_compact -B build -DPython_EXECUTABLE="$PWD/venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release > cmake.log
cmake --build build -j 16 > build.log
PYTHONPATH=build OMP_NUM_THREADS=16 venv/bin/python src/benchmark/experiments/issue33_compact/run.py results.csv \
  gemini="$data/gemini:768" openai="$data/openai:1536" | tee run.log
