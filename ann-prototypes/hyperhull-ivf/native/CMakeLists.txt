cmake_minimum_required(VERSION 3.26)
project(hyperhull_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # the static TieredQ library is linked into a Python module

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# TieredQ repository root (this project lives in ann-prototypes/hyperhull-ivf/native)
set(TIEREDQ_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH "TieredQ source tree")
add_subdirectory(${TIEREDQ_ROOT} tieredq EXCLUDE_FROM_ALL)

nanobind_add_module(_hyperhull NOMINSIZE cpp/bindings.cpp cpp/router.cpp cpp/partition.cpp)
target_compile_options(_hyperhull PRIVATE -O3 -g -march=native -Wall -Wextra)
# partition.cpp uses TieredQ's e-means trainer and BLAS assignment, which are internal headers under src/
target_include_directories(_hyperhull PRIVATE ${TIEREDQ_ROOT}/src)
# ... and is compiled like the TieredQ library (runtime SIMD dispatch, no FMA contraction), so the inline trainer code
# it shares with the library is identical in both
set_source_files_properties(cpp/partition.cpp PROPERTIES COMPILE_OPTIONS "-march=x86-64;-ffp-contract=off")
target_link_libraries(_hyperhull PRIVATE TieredQ::TieredQ)
install(TARGETS _hyperhull LIBRARY DESTINATION hyperhull_native)
