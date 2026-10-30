# Builds the issue #6 IVF eval against one TieredQ tree and the benchmark's RaBitQ overlay, with the benchmark's
# shared optimization policy: cmake -S . -B out -DTIEREDQ_SOURCE=<tree> -DRABITQ_SOURCE=<tree>/benchmark/.deps/rabitq-native
cmake_minimum_required(VERSION 3.18)
project(issue6_ivf CXX)
set(CMAKE_CXX_STANDARD 20)
add_compile_options(-O3 -march=native -ffp-contract=off)
set(TIEREDQ_TESTS OFF CACHE BOOL "" FORCE)
set(TIEREDQ_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${TIEREDQ_SOURCE} tieredq)
find_package(OpenMP REQUIRED)
add_executable(issue6_ivf_eval ivf_eval.cpp)
target_include_directories(issue6_ivf_eval PRIVATE "${RABITQ_SOURCE}/include")
target_link_libraries(issue6_ivf_eval PRIVATE TieredQ::TieredQ OpenMP::OpenMP_CXX)
