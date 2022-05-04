# Minimum CMake version 3.18 for CUDA --std=c++17 and findCUDAToolkit
cmake_minimum_required(VERSION VERSION 3.18 FATAL_ERROR)

project(cuRVE LANGUAGES CXX CUDA)

# Define include and source files/dirs
get_filename_component(CURVE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include REALPATH)
get_filename_component(CURVE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} REALPATH)

SET(SRC_INCLUDE
    ${CURVE_INCLUDE_DIR}/curve/curve.cuh
)
SET(SRC_CURVE
    ${CURVE_SRC_DIR}/curve/curve.cu
)
SET(ALL_SRC
    ${SRC_INCLUDE}
    ${SRC_CURVE}
)

# Use the FindCUDATooklit package (CMake > 3.17) to find other parts of the cuda toolkit not provided by the CMake language support
include(FindCUDAToolkit)
find_package(CUDAToolkit REQUIRED)

# Policy to enable use of separate device link options, introduced in CMake 3.18
cmake_policy(SET CMP0105 NEW)

# Add custom modules directory
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/../cmake/modules/ ${CMAKE_MODULE_PATH})

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

# Define output location of static library
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # If top level project
    SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE}/)
else()
    # If called via add_subdirectory()
    SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../lib/${CMAKE_BUILD_TYPE}/)
endif()

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

# Define the list of sources 

# Add the executable
# Define which source files are required for the target executable
add_library(${PROJECT_NAME} STATIC ${ALL_SRC})

# Add the library headers as public so they are forwarded on.
target_include_directories(${PROJECT_NAME} PUBLIC "${CURVE_INCLUDE_DIR}")
# Add any private headers.
target_include_directories(${PROJECT_NAME} PRIVATE "${CURVE_SRC_DIR}")

# enable "fpic" for linux to allow shared libraries to be build from the static library (required for swig)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

# Enable RDC for the target
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link against dl if available. This used to be implied from somewhere...
if(CMAKE_DL_LIBS)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${CMAKE_DL_LIBS})
endif()

# Set target level warnings.
EnableFLAMEGPUCompilerWarnings(TARGET "${PROJECT_NAME}")
# Apply common compiler settings
CommonCompilerSettings(TARGET "${PROJECT_NAME}")
# Set the cuda gencodes, potentially using the user-provided CUDA_ARCH
SetCUDAGencodes(TARGET "${PROJECT_NAME}")

target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)

if(USE_NVTX AND TARGET NVTX::nvtx)
    target_link_libraries(${PROJECT_NAME} PUBLIC NVTX::nvtx)
    # Get the version to set a definition value 
    # Interface targets only support the version property from 3.19
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.19")
        get_target_property(nvtxversion NVTX::nvtx VERSION)
    else()
        # fallback to the old cmake var.
        set(nvtxversion ${NVTX_VERSION})
    endif()
    target_compile_definitions(${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANGUAGE:C,CXX,CUDA>:USE_NVTX=${nvtxversion}>")
    unset(nvtxversion)
endif()

# If gcc, need to add linker flag for std::experimental::filesystem pre c++17
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(${PROJECT_NAME} PUBLIC stdc++fs)
endif()

