include_guard(DIRECTORY)

# Policy to enable use of separate device link options, introduced in CMake 3.18
cmake_policy(SET CMP0105 NEW)

# Common rules for other cmake files
# Don't create installation scripts (and hide CMAKE_INSTALL_PREFIX from cmake-gui)
set(CMAKE_SKIP_INSTALL_RULES TRUE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE INTERNAL "" FORCE)
# Option to promote compilation warnings to error, useful for strict CI
option(WARNINGS_AS_ERRORS "Promote compilation warnings to errors" OFF)
# Option to group CMake generated projects into folders in supported IDEs
option(CMAKE_USE_FOLDERS "Enable folder grouping of projects in IDEs." ON)
mark_as_advanced(CMAKE_USE_FOLDERS)

# Set a default build type if not passed
get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(${GENERATOR_IS_MULTI_CONFIG})
    # CMAKE_CONFIGURATION_TYPES defaults to something platform specific
    # Therefore can't detect if user has changed value and not reset it
    # So force "Debug;Release"
    # set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE INTERNAL
        # "Choose the types of build, options are: Debug Release." FORCE)#
else()
    if(NOT CMAKE_BUILD_TYPE)
        set(default_build_type "Release")
        message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
        set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING 
            "Choose the type of build, options are: Release, Debug, RelWithDebInfo, MinSizeRel or leave the value empty for the default." FORCE)
    endif()
endif()

# Ask Cmake to output compile_commands.json (if supported). This is useful for vscode include paths, clang-tidy/clang-format etc
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "Control the output of compile_commands.json")

# Define a function which can be used to set common compiler options for a target
# We do not want to force these options on end users (although they should be used ideally), hence not just public properties on the library target
# Function to suppress compiler warnings for a given target
function(CommonCompilerSettings)
    # Parse the expected arguments, prefixing variables.
    cmake_parse_arguments(
        CCS
        ""
        "TARGET"
        ""
        ${ARGN}
    )

    # Ensure that a target has been passed, and that it is a valid target.
    if(NOT CCS_TARGET)
        message( FATAL_ERROR "function(CommonCompilerSettings): 'TARGET' argument required")
    elseif(NOT TARGET ${CCS_TARGET} )
        message( FATAL_ERROR "function(CommonCompilerSettings): TARGET '${CCS_TARGET}' is not a valid target")
    endif()

    # Add device debugging symbols to device builds of CUDA objects
    target_compile_options(${CCS_TARGET} PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>")
    # Ensure DEBUG and _DEBUG are defined for Debug builds
    target_compile_definitions(${CCS_TARGET} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:DEBUG>)
    target_compile_definitions(${CCS_TARGET} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:_DEBUG>)
    # Enable -lineinfo for Release builds, for improved profiling output.
    target_compile_options(${CCS_TARGET} PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-lineinfo>")

    # Set an NVCC flag which allows host constexpr to be used on the device.
    target_compile_options(${CCS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

    # Prevent windows.h from defining max and min.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_definitions(${CCS_TARGET} PRIVATE NOMINMAX)
    endif()

    # Pass the SEATBELTS macro, which when set to off/0 (for non debug builds) removes expensive operations.
    if (SEATBELTS)
        # If on, all build configs have  seatbelts
        target_compile_definitions(${CCS_TARGET} PRIVATE SEATBELTS=1)
    else()
        # Id off, debug builds have seatbelts, non debug builds do not.
        target_compile_definitions(${CCS_TARGET} PRIVATE $<IF:$<CONFIG:Debug>,SEATBELTS=1,SEATBELTS=0>)
    endif()

    # MSVC handling of SYSTEM for external includes.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.10)
        # These flags don't currently have any effect on how CMake passes system-private includes to msvc (VS 2017+)
        set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "/external:I")
        set(CMAKE_INCLUDE_SYSTEM_FLAG_CUDA "/external:I")
        # VS 2017+
        target_compile_options(${CCS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/experimental:external>")
    endif()

    # Enable parallel compilation
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${CCS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /MP>")
        target_compile_options(${CCS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/MP>")
    endif()

    # If CUDA 11.2+, can build multiple architectures in parallel. 
    # Note this will be multiplicative against the number of threads launched for parallel cmake build, which may lead to processes being killed, or excessive memory being consumed.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.2" AND USE_NVCC_THREADS AND DEFINED NVCC_THREADS AND NVCC_THREADS GREATER_EQUAL 0)
        target_compile_options(${CCS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--threads ${NVCC_THREADS}>")
    endif()

endfunction()