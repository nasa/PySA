cmake_minimum_required(VERSION 3.26)
project(pysa_cdcl)
include_directories(include)
set(CMAKE_CXX_STANDARD 20)
if(SATTRACE)
    add_compile_definitions("SATTRACE")
endif ()
add_library(cdcl_lib STATIC cdcl.cpp)
add_executable(test_cdcl ./tests/cdcl_tests.cpp cdcl.cpp)
add_executable(pysa_cdcl main.cpp cdcl.cpp)

## Python bindings

# Use FindPython or FindPython3 module
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)

# Check for pybind11
find_package(pybind11 REQUIRED)

# Build the Python extension bindings module
pybind11_add_module(pysa_cdcl_ext bindings.cpp)
target_link_libraries(pysa_cdcl_ext PUBLIC cdcl_lib)
# make sure object name matches the defined module name
set_target_properties(pysa_cdcl_ext PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pysa2_cdcl
        OUTPUT_NAME "bindings")
# Installs only the Python extension object when the "bindings" component is specified in setup.py
install(
        TARGETS pysa_cdcl_ext
        DESTINATION ${CMAKE_INSTALL_PREFIX}
        COMPONENT bindings
)