#!/bin/sh
# C++ and Python test suites of the e-means-default tree (source-defaults), 32 threads.
cd /home/sean/experiments/TieredQ-issue45-emeans-stream-20261118/source-defaults
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release > ../cmake.log 2>&1 && cmake --build . -j 32 --target test_tieredq > ../make.log 2>&1 || { echo BUILD FAILED; tail -20 ../make.log; exit 1; }
OMP_NUM_THREADS=32 ./test_tieredq > ../ctest.log 2>&1; echo "cpp exit $?"; grep -iE "fail|error|^OK" ../ctest.log | head
cd .. && CMAKE_BUILD_PARALLEL_LEVEL=32 uv run --with pytest python -m pytest -q tests/python > pytest.log 2>&1; echo "py exit $?"; tail -5 pytest.log
