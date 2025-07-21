# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

cmake_minimum_required(VERSION 3.28)
project(
  "${SKBUILD_PROJECT_NAME}"
  LANGUAGES CXX
  VERSION "${SKBUILD_PROJECT_VERSION}")

include(FetchContent)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(ABSL_PROPAGATE_CXX_STD ON)

# Remove support for scan deps, which is only useful when using C++ modules.
unset(CMAKE_CXX_SCANDEP_SOURCE)

# Use local submodules from the 'include' directory and configure them
# for a clean offline build by disabling all test suites and external fetching.

# For abseil-cpp, disable its testing to prevent gtest download.
set(ABSL_BUILD_TESTING OFF CACHE BOOL "Disable Abseil testing" FORCE)
add_subdirectory(include/abseil-cpp)

# For pybind11, disable its tests.
set(PYBIND11_TEST OFF CACHE BOOL "Disable pybind11 testing" FORCE)
add_subdirectory(include/pybind11)

# For pybind11_abseil, it has no tests to disable, but we link it.
add_subdirectory(include/pybind11_abseil)

# For cifpp, disable its tests and its own data download mechanism.
set(BUILD_TESTING OFF CACHE BOOL "Disable libcifpp testing" FORCE)
set(CIFPP_DOWNLOAD_CCD OFF CACHE BOOL "Disable cifpp CCD download" FORCE)
add_subdirectory(include/libcifpp)

# For dssp, disable its tests and its own dependency fetching.
set(DSSP_FETCH_SUPPORT_LIBS OFF CACHE BOOL "Disable dssp support lib fetching" FORCE)
set(DSSP_BUILD_TESTS OFF CACHE BOOL "Disable dssp testing" FORCE)
add_subdirectory(include/dssp)

find_package(
  Python3
  COMPONENTS Interpreter Development NumPy
  REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(src/)

file(GLOB_RECURSE cpp_srcs src/alphafold3/*.cc)
list(FILTER cpp_srcs EXCLUDE REGEX ".*\(_test\|_main\|_benchmark\).cc$")

add_compile_definitions(NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

pybind11_add_module(cpp ${cpp_srcs})

target_link_libraries(
  cpp
  PRIVATE absl::check
          absl::flat_hash_map
          absl::node_hash_map
          absl::strings
          absl::status
          absl::statusor
          absl::log
          pybind11_abseil::absl_casters
          Python3::NumPy
          dssp::dssp
          cifpp::cifpp)

target_compile_definitions(cpp PRIVATE VERSION_INFO=${PROJECT_VERSION})
install(TARGETS cpp LIBRARY DESTINATION alphafold3)
install(
  FILES LICENSE
        OUTPUT_TERMS_OF_USE.md
        WEIGHTS_PROHIBITED_USE_POLICY.md
        WEIGHTS_TERMS_OF_USE.md
  DESTINATION alphafold3)
