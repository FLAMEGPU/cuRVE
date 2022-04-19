# cuRVE : The CUDA Runtime Variable Environment

## Overview

The CUDA Runtime Variable Environment (curRVE) is a library which provides key-value memory management and access for CUDA global device memory.

GPU device memory can be registered and values set in host code using a constant string expression key via the the cuRVE API. For example the following initialises and sets the value of three cuRVE variables.

```cuda
curveInit(VECTOR_ELEMENTS);
curveRegisterVariable("a");
curveRegisterVariable("b");
curveRegisterVariable("c");

for (int i=0; i<VECTOR_ELEMENTS; i++){
    float a = rand()/(float)RAND_MAX;
    float b = rand()/(float)RAND_MAX;
    curveSetFloat("a", i, a);
    curveSetFloat("b", i, b);
}
```

A corresponding kernel can be defined using the cuRVE variable access functions as follows;

```cuda
__global__ void vectorAdd()
{
    float a, b, c;
    
    unsigned int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
     
    a = getFloatVariable("a", idx);
    b = getFloatVariable("b", idx);
    c = a + b;
    setFloatVariable("c", c, idx);
}
```

## Namespaces

Namespaces allow a cuRVE variable to have a limited scope. Similarly, variable names can be re-used under different namespaces. The cuRVE namespace can be changed using  an API call as follows;

```cuda
curveChangeNamespace("vector_addition_example");
```

Namespaces can be used to limit a variable to a particular CUDA kernel (or set of kernels). Further restrictions on variable access may be placed by enabling or disabling variable access as follows;

```cuda
curveDisableVariable("a");
curveEnableVariable("b");
```

## Error Checking

Errors can occur on the device or host and error codes can be obtained by using the following API functions;

```cuda
curveGetLastHostError();    // Host API function which gets the last host API error code
curveGetLastDeviceError();  // Device API function which gets the last device API error code
```

Formatted errors can be output using the following API calls which will use the current source file, function name and line number;

```cuda
curveReportErrors();      //Host API function outputs any host or device errors to std:out
curveReportHostError();   //Host API function outputs the last host API error
curveReportDeviceError(); //Device API function outputs the last device API error
```

## Documentation

The cuRVE header file is commented using doxygen format comments.

## Building the Example

cuRVE uses [CMake](https://cmake.org/), as a cross-platform process, for configuring and generating build directives, e.g. `Makefile` or `.vcxproj`.
This is used to build the cuRVE library and example(s).

### Requirements

+ [CMake](https://cmake.org/download/) `>= 3.18`
+ [CUDA](https://developer.nvidia.com/cuda-downloads) `>= 11.0` and a [Compute Capability](https://developer.nvidia.com/cuda-gpus) `>= 3.5` NVIDIA GPU.
+ C++17 capable C++ compiler (host), compatible with the installed CUDA version
  + [Microsoft Visual Studio 2019](https://visualstudio.microsoft.com/) (Windows)
    + *Note:* Visual Studio must be installed before the CUDA toolkit is installed. See the [CUDA installation guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) for more information.
  + [make](https://www.gnu.org/software/make/) and [GCC](https://gcc.gnu.org/) `>= 7` (Linux)
  + Older C++ compilers which support C++14 may currently work, but support will be dropped in a future release.

### Building with CMake

Building via CMake is a three step process, with slight differences depending on your platform.

1. Create a build directory for an out-of tree build
2. Configure CMake into the build directory
    + Using the CMake GUI or CLI tools
    + Specifying build options such as the CUDA Compute Capabilities to target, or the use of NVTX markers for profiling. See [CMake Configuration Options](#CMake-Configuration-Options) for details of the available configuration options
3. Build compilation targets using the configured build system
    + See [Available Targets](#Available-targets) for a list of available targets.

#### Linux

To build under Linux using the command line, you can perform the following steps.

For example, to configure CMake for `Release` builds, for consumer Pascal GPUs (Compute Capability `61`), with python bindings enabled, producing the static library and `boids_bruteforce` example binary.

```bash
# Create the build directory and change into it
mkdir -p build && cd build

# Configure CMake from the command line passing configure-time options. 
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=61 -DBUILD_SWIG_PYTHON=ON

# Build the required targets. In this case all targets
cmake --build . --target flamegpu boids_bruteforce -j 8

# Alternatively make can be invoked directly
make flamegpu boids_bruteforce -j8

```

#### Windows

Under Windows, you must instruct CMake on which Visual Studio and architecture to build for, using the CMake `-A` and `-G` options.
This can be done through the GUI or the CLI.

I.e. to configure CMake for consumer Pascal GPUs (Compute Capability `61`), with python bindings enabled, and build the producing the static library and `boids_bruteforce` example binary in the Release configuration:

```cmd
REM Create the build directory 
mkdir build
cd build

REM Configure CMake from the command line, specifying the -A and -G options. Alternatively use the GUI
cmake .. -A x64 -G "Visual Studio 16 2019" -DCUDA_ARCH=61 -DBUILD_SWIG_PYTHON=ON

REM You can then open Visual Studio manually from the .sln file, or via:
cmake --open . 
REM Alternatively, build from the command line specifying the build configuration
cmake --build . --config Release --target flamegpu boids_bruteforce --verbose
```

#### CMake Configuration Options

| Option                   | Value                                           | Description                                                                                                                                          |
| ------------------------ | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CMAKE_BUILD_TYPE`       | `Release`/`Debug`/`MinSizeRel`/`RelWithDebInfo` | Select the build configuration for single-target generators such as `make`                                                                           |
| `CUDA_ARCH`              | `"52 60 70 80"`                                 | Select [CUDA Compute Capabilities](https://developer.nvidia.com/cuda-gpus) to build/optimise for, as a space or `;` separated list. Defaults to `""` |
| `USE_NVTX`               | `ON`/`OFF`                                      | Enable NVTX markers for improved profiling. Default `OFF`                                                                                            |
| `WARNINGS_AS_ERRORS`     | `ON`/`OFF`                                      | Promote compiler/tool warnings to errors are build time. Default `OFF`                                                                               |

For a list of available CMake configuration options, run the following from the `build` directory:

```bash
cmake -LH ..
```

#### Available Targets

| Target         | Description                                    |
| -------------- | -----------------------------------------------|
| `all`          | Linux target containing default set of targets |
| `ALL_BUILD`    | The windows/MSVC equivalent of `all`           |
| `cuRVE`        | Build cuRVE static library                     |
| `example`      | Build `example` demonstrating use of cuRVE     |

For a full list of available targets, run the following after configuring CMake:

```bash
cmake --build . --target help
```

## Executing the example

Once compiled the example binary can be executed, to demonstrate use and performance of cuRVE

I.e. for a `Release` build of the `example`, run:

```bash
./bin/Release/example --help
```
