# Minimum CMake version 3.18 for CUDA --std=c++17 and findCUDAToolkit
cmake_minimum_required(VERSION VERSION 3.18 FATAL_ERROR)

project(cuRVE LANGUAGES CXX CUDA)

include(FindCUDAToolkit)
find_package(CUDAToolkit)

# Ensure this is not an in-source build
include(./cmake/OutOfSourceOnly.cmake)

# Check compiler functionailty, as there are known issues in some cases, but version checks are not always sufficient.
include(./cmake/CheckCompilerFunctionality.cmake)
# If this returned a negative result, set the docs only build. 
if(NOT CheckCompilerFunctionality_RESULT)
    message(FATAL_ERROR "Unsupported Compiler Combination. See prior warnings for details")
endif()

# Don't create installation scripts (and hide CMAKE_INSTALL_PREFIX from cmake-gui)
set(CMAKE_SKIP_INSTALL_RULES TRUE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE INTERNAL ""  FORCE)

# Supports CUDA 11.0 and newer for C++17?
set(MINIMUM_CUDA_VERSION 11.0)
# If the CUDA compiler is too old, trigger a docs only build.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS ${MINIMUM_CUDA_VERSION})
    set(DOCUMENTATION_ONLY_BUILD ON)
    message(STATUS "Documentation-only build: CUDA ${MINIMUM_SUPPORTED_CUDA_VERSION} or greater is required for compilation.")
endif()


# Option to enable/disable NVTX markers for improved profiling
option(USE_NVTX "Build with NVTX markers enabled" ON)

# Control target CUDA_ARCH to compile for. This could be migrated to the CMake > 3.18 way of doing this (exluding LTO support)
SET(CUDA_ARCH "${CUDA_ARCH}" CACHE STRING "List of CUDA Architectures to target. E.g. 61;70" FORCE)

# Add the library building subdirectory
add_subdirectory(src "${PROJECT_BINARY_DIR}/cuRVE")
# Set as startup project
set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"  PROPERTY VS_STARTUP_PROJECT cuRVE)

# Add the example as a separate subdirectory
add_subdirectory(example)
