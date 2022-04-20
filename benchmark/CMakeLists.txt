# Minimum CMake version 3.18 for CUDA --std=c++17 and findCUDAToolkit
cmake_minimum_required(VERSION VERSION 3.18 FATAL_ERROR)

project(benchmark LANGUAGES CXX CUDA)

# Define the list of sources 
# Define include and source files/dirs
get_filename_component(BENCHMARK_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} REALPATH)
get_filename_component(BENCHMARK_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} REALPATH)

SET(SRC_INCLUDE
    ${BENCHMARK_INCLUDE_DIR}/util/nvtx.h
    ${BENCHMARK_INCLUDE_DIR}/util/CUDAErrorChecking.cuh
    ${BENCHMARK_INCLUDE_DIR}/util/Timer.h
    ${BENCHMARK_INCLUDE_DIR}/util/SteadyClockTimer.h
    ${BENCHMARK_INCLUDE_DIR}/util/CUDAEventTimer.cuh
    ${BENCHMARK_INCLUDE_DIR}/util/wddm.cuh
)
SET(SRC_BENCHMARK
    ${BENCHMARK_SRC_DIR}/util/wddm.cu
    ${BENCHMARK_SRC_DIR}/main.cu
)
SET(ALL_SRC
    ${SRC_INCLUDE}
    ${SRC_BENCHMARK}
)

# Use the FindCUDATooklit package (CMake > 3.17) to find other parts of the cuda toolkit not provided by the CMake language support
include(FindCUDAToolkit)
find_package(CUDAToolkit REQUIRED)

# Policy to enable use of separate device link options, introduced in CMake 3.18
cmake_policy(SET CMP0105 NEW)

# Add custom modules directory
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/../cmake/modules/ ${CMAKE_MODULE_PATH})

# Define output location of binary files
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # If top level project
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/)
else()
    # If called via add_subdirectory()
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../bin/${CMAKE_BUILD_TYPE}/)
endif()

# Ensure this is not an in-source build
# This might be a little aggressive to go in comon.
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/OutOfSourceOnly.cmake)

# include CMake function to enable setting of warning suppressions
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/warnings.cmake)
# include CMake function to enable setting of gencode flags
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/cuda_arch.cmake)
# Set the C++ and CUDA standard to use
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/cxxstd.cmake)
# Include the common rules
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/common.cmake)

# Fetch 3rd party dependencies and make them available
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/dependencies/CLI11.cmake)

# Control how we link against the cuda runtime library (CMake >= 3.17)
# We may wish to use static or none instead, subject to python library handling.
set(CMAKE_CUDA_RUNTIME_LIBRARY shared)

# If NVTX is enabled, find the library and update variables accordingly. 
if(USE_NVTX)
    # Find the nvtx library using custom cmake module, providing imported targets
    # Do not use CUDA::nvToolsExt as this always uses NVTX1 not 3.
    find_package(NVTX)
    # If the targets were not found, emit a warning 
    if(NOT TARGET NVTX::nvtx)
        # If not found, emit a warning and continue without NVTX
        message(WARNING "NVTX could not be found. Proceeding with USE_NVTX=OFF")
        if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
            SET(USE_NVTX "OFF" PARENT_SCOPE)
        endif()
    endif()
endif(USE_NVTX)


# Add the executable
add_executable(${PROJECT_NAME} ${ALL_SRC})

# Set target level warnings.
EnableFLAMEGPUCompilerWarnings(TARGET "${PROJECT_NAME}")
# Apply common compiler settings
CommonCompilerSettings(TARGET "${PROJECT_NAME}")
# Set the cuda gencodes, potentially using the user-provided CUDA_ARCH
SetCUDAGencodes(TARGET "${PROJECT_NAME}")

# Enable RDC for the target
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link against the cxxopts cli library
# @todo - this is not an imported target, so the implicit include directory is not treated as SYSTEM. Can either modify the target, or potentially trigger an install to generate the config then find package instead? 
target_link_libraries(${PROJECT_NAME} PRIVATE CLI11::CLI11)

# Link against the flamegpu static library target.
target_link_libraries(${PROJECT_NAME} PRIVATE cuRVE)
# Workaround for incremental rebuilds on MSVC, where device link was not being performed.
# https://github.com/FLAMEGPU/FLAMEGPU2/issues/483
if(MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.1")
    # Provide the absolute path to the lib file, rather than the relative version cmake provides.
    target_link_libraries(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE:curve>")
endif()
