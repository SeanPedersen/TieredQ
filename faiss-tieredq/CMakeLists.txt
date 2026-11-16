# faiss-tieredq (#55): IndexIVFTieredQ, a Faiss IVF index on TieredQ fast-scan tiles, built against an installed Faiss
# (v1.16.0; CMAKE_PREFIX_PATH must find faiss-config.cmake) and the TieredQ tree it ships in.
cmake_minimum_required(VERSION 3.24)
project(faiss_tieredq VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(FAISS_TIEREDQ_STANDALONE OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(FAISS_TIEREDQ_STANDALONE ON)
endif()
option(FAISS_TIEREDQ_TESTS "Build the faiss-tieredq tests" ${FAISS_TIEREDQ_STANDALONE})
option(FAISS_TIEREDQ_BENCHMARKS "Build the acceptance benchmark" OFF)
# faiss (generic), faiss_avx2 or faiss_avx512: the SIMD build of Faiss's own kernels to link.
set(FAISS_TIEREDQ_FAISS_TARGET "faiss" CACHE STRING "Faiss target to link")
set(TIEREDQ_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "TieredQ source tree")

find_package(faiss 1.16 REQUIRED)
find_package(OpenMP REQUIRED)
if(NOT TARGET TieredQ::TieredQ)
  set(TIEREDQ_TESTS OFF CACHE BOOL "" FORCE)
  set(TIEREDQ_EXAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory(${TIEREDQ_SOURCE_DIR} tieredq EXCLUDE_FROM_ALL)
endif()

add_library(faiss_tieredq STATIC src/IndexIVFTieredQ.cpp src/index_io.cpp)
set_target_properties(faiss_tieredq PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Faiss exports its include directory on the generic target only; code that includes Faiss's headers is compiled with
# the ISA of the Faiss variant it links (Faiss's own FAISS_OPT_LEVEL flags). TieredQ keeps its runtime dispatch.
target_include_directories(faiss_tieredq PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                           $<TARGET_PROPERTY:faiss,INTERFACE_INCLUDE_DIRECTORIES>)
if(FAISS_TIEREDQ_FAISS_TARGET STREQUAL "faiss_avx512")
  target_compile_options(faiss_tieredq PUBLIC -mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt)
elseif(FAISS_TIEREDQ_FAISS_TARGET STREQUAL "faiss_avx2")
  target_compile_options(faiss_tieredq PUBLIC -mavx2 -mfma -mf16c -mpopcnt)
endif()
target_compile_options(faiss_tieredq PRIVATE -O3 -Wall -Wextra)
target_link_libraries(faiss_tieredq PUBLIC ${FAISS_TIEREDQ_FAISS_TARGET} TieredQ::TieredQ OpenMP::OpenMP_CXX)

if(FAISS_TIEREDQ_TESTS)
  enable_testing()
  add_executable(test_faiss_tieredq tests/test_index_ivf_tieredq.cpp)
  target_link_libraries(test_faiss_tieredq PRIVATE faiss_tieredq)
  add_test(NAME test_faiss_tieredq COMMAND test_faiss_tieredq)
endif()

if(FAISS_TIEREDQ_BENCHMARKS)
  add_executable(faiss_tieredq_acceptance benchmarks/acceptance.cpp)
  target_compile_options(faiss_tieredq_acceptance PRIVATE -O3)
  target_link_libraries(faiss_tieredq_acceptance PRIVATE faiss_tieredq)
endif()

if(SKBUILD)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE
                  OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_core NB_STATIC python/module.cpp)
  target_link_libraries(_core PRIVATE faiss_tieredq)
  install(TARGETS _core LIBRARY DESTINATION faiss_tieredq)
endif()
