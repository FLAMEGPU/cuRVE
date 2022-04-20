# cuRVE ABM-like Benchmark

This directory contains a cuRVE benchmark application.

It is an ABM-like application, mocking the usage pattern of cuRVE within [FLAME GPU 2](https://github.com/FLAMEGPU/FLAMEGPU2).

## Directory Structure

+ `CMakeLists.txt` contains the main CMakeLists file for this individual benchmark. 
  + It is expected to be configured as part of the parent CMake project and may not work in isolation
+ `main.cu` contains the main body of the benchmark model
+ `util/` contains a number of CUDA / C++ classes / namespaced functions from the main FLAMEGPU/FLAMEGPU2 repository to simplify `main.cu` by abstracting NVTX, Timing and common CUDA usage in isolation. These are generally unimportant in terms of benchmark content

## Building the Benchmark

From the repository root, create a `build` directory, configure `CMake` and build the executable. See the main [Readme.md](../README.md) for more detail.

```bash
# Cd to the repo root
cd ../
# Create a build directory
mkdir -p build && cd build
# Configure CMake, in this case for consume Ampere GPUs (SM_86)
cmake .. -DCUDA_ARCH=86 -DUSE_NVTX=ON
# Build the binary using 8 parallel jobs
cmake --build . --target benchmark -j 8
```

## Running the Benchmark

Usage of the binary is currently evolving, for a current build, use `-h/--help` to list available arguments.

```bash
# From the repo root, for a Release configuration
./bin/Release/benchmark --help
```
