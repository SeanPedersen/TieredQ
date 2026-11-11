# Optional Python extension; standalone C++ builds never discover Python.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE NANOBIND_STATUS)
if(NOT NANOBIND_STATUS EQUAL 0)
  message(FATAL_ERROR "Cannot locate nanobind for the selected Python interpreter")
endif()
find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(_core NB_STATIC bindings/module.cpp bindings/index.cpp bindings/lists.cpp bindings/ivf.cpp bindings/nested.cpp)
target_link_libraries(_core PRIVATE TieredQ::TieredQ)
install(TARGETS _core LIBRARY DESTINATION tieredq COMPONENT python)
install(FILES LICENSE NOTICE DESTINATION tieredq/licenses COMPONENT python)

install(DIRECTORY licenses/ DESTINATION tieredq/licenses COMPONENT python)
