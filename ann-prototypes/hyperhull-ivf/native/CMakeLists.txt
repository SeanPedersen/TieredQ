cmake_minimum_required(VERSION 3.26)
project(hyperhull_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_hyperhull NOMINSIZE cpp/bindings.cpp cpp/router.cpp)
target_compile_options(_hyperhull PRIVATE -O3 -g -march=native -Wall -Wextra)
install(TARGETS _hyperhull LIBRARY DESTINATION hyperhull_native)
