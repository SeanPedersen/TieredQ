#!/bin/bash
set -e
cd "$(dirname "$0")"
source venv/bin/activate
cmake -S faiss-main -B build -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DBUILD_TESTING=OFF \
  -DFAISS_OPT_LEVEL=avx512 -DBLA_VENDOR=OpenBLAS -DPython_EXECUTABLE=$PWD/venv/bin/python > configure.log 2>&1
cmake --build build -j 32 --target swigfaiss_avx512 > build.log 2>&1
(cd build/faiss/python && ../../../venv/bin/python -m pip --version >/dev/null 2>&1 || true; ../../../venv/bin/python setup.py build > ../../../setup.log 2>&1)
echo done
