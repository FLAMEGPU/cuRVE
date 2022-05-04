# Fetch and make the BSD Licensed https://github.com/CLIUtils/CLI11 cli parsing library available for use via cmake

include(FetchContent)

# Declare the content to fetc
FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.2.0
)
# Get the properties
FetchContent_GetProperties(cli11)
# If not yet populated
if(NOT cli11_POPULATED)
    # Populate
    FetchContent_Populate(cli11)
    set(CLI11_INSTALL OFF)
    set(CLI11_BUILD_TESTS OFF)
    set(CLI11_BUILD_EXAMPLES OFF)
    set(CLI11_BUILD_DOCS OFF)
    # Add the cmake-based project
    add_subdirectory(${cli11_SOURCE_DIR} ${cli11_BINARY_DIR})
    unset(CLI11_INSTALL)
    unset(CLI11_BUILD_TESTS)
    unset(CLI11_BUILD_EXAMPLES)
    unset(CLI11_BUILD_DOCS)
endif()

# Verify that CLI11 is available
if (NOT TARGET CLI11::CLI11)
    message(FATAL_ERROR "CLI11::CLI11 has not been fetched correctly")
endif()