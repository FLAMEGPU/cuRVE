/**
 * cuRVE read-write benchmark
 * 
 * This contains a benchmark of cuRVE for reading and writing from/to global memory  
 * 
 * @todo list:
 * + Add erorr checking
 * + Add validation
 * + Clean up / refactor 
 *   + MockSimulation class to reduce repeatedly passed parameters
 *   + Split out curve mapping / unmapping
 * + Expand mock agent pop / mock 
 * + Standalone curve tests outside of this?
 * + Add reference non-curve implementation for performance comparison
 *   + Unlikely in the immediate future
 */

#include <cstdio>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <random>

#include "cuda_runtime.h"

// CLI library
#include <CLI/CLI.hpp>

// Include a number of utilty classes to simplify code in the benchmark itself
#include "util/Timer.h"
#include "util/SteadyClockTimer.h"
#include "util/CUDAErrorChecking.cuh"
#include "util/CUDAEventTimer.cuh"
#include "util/wddm.cuh"
#include "util/nvtx.h"
#include "util/OutputCSV.hpp"

// Include the curve header
#include "curve/curve.cuh"

// Anonymous namespace for locally scoped global state
namespace {
    // file-scope only variable used to cache the driver mode
    bool deviceUsingWDDM = false;
    // Inlined method in the anonymous namespace to create a new timer, subject to the driver model.
    std::unique_ptr<util::Timer> getDriverAppropriateTimer() {
        if (!deviceUsingWDDM) {
            return std::unique_ptr<util::Timer>(new util::CUDAEventTimer());
        } else {
            return std::unique_ptr<util::Timer>(new util::SteadyClockTimer());
        }
    }
}  // anonymous namespace

#define AGENT_NAME "agent"

#define MULTIPLIER 1

/**
 * Class to mock a population of agents, storing agent data SoA-like. This is only used on the host
 */
class HD_DATA {
 public:
    uint32_t length = 0;

    uint32_t * h_u32 = nullptr;
    float * h_f = nullptr;
    double * h_d = nullptr;

    uint32_t * d_u32 = nullptr;
    float * d_f = nullptr;
    double * d_d = nullptr;

    HD_DATA(const uint32_t length) :
        length(0),
        h_u32(nullptr),
        h_f(nullptr),
        h_d(nullptr),
        d_u32(nullptr),
        d_f(nullptr),
        d_d(nullptr) {
        this->allocate(length);
    }

    ~HD_DATA() {
        this->deallocate();
    }

    void allocate(const uint32_t length) {
        NVTX_RANGE(__func__);
        if (length != 0) {
            this->deallocate();
        }
        this->length = length;
        this->h_u32 = static_cast<uint32_t*>(std::malloc(length * sizeof(uint32_t)));
        this->h_f = static_cast<float*>(std::malloc(length * sizeof(float)));
        this->h_d = static_cast<double*>(std::malloc(length * sizeof(double)));
        memset(this->h_u32, 0, length * sizeof(uint32_t));
        memset(this->h_f, 0, length * sizeof(float));
        memset(this->h_d, 0, length * sizeof(double));

        gpuErrchk(cudaMalloc(&this->d_u32, length * sizeof(uint32_t)));
        gpuErrchk(cudaMalloc(&this->d_f, length * sizeof(float)));
        gpuErrchk(cudaMalloc(&this->d_d, length * sizeof(double)));

        gpuErrchk(cudaMemset(this->d_u32, 0, length * sizeof(uint32_t)));
        gpuErrchk(cudaMemset(this->d_f, 0, length * sizeof(float)));
        gpuErrchk(cudaMemset(this->d_d, 0, length * sizeof(double)));
    };

    void deallocate() {
        NVTX_RANGE(__func__);
        this->length = 0;
        if(this->h_u32 != nullptr) {
            std::free(this->h_u32);
            this->h_u32 = nullptr;
        }
        if(this->h_f != nullptr) {
            std::free(this->h_f);
            this->h_f = nullptr;
        }
        if(this->h_d != nullptr) {
            std::free(this->h_d);
            this->h_d = nullptr;
        }
        if(this->d_f != nullptr) {
            gpuErrchk(cudaFree(this->d_f));
            this->d_f = nullptr;
        }
        if(this->d_d != nullptr) {
            gpuErrchk(cudaFree(this->d_d));
            this->d_d = nullptr;
        }
    };

    void copyh2d() {
        NVTX_RANGE(__func__);
        if(length > 0) {
            gpuErrchk(cudaMemcpy(d_u32, h_u32, length * sizeof(uint32_t), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_f, h_f, length * sizeof(float), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_d, h_d, length * sizeof(double), cudaMemcpyHostToDevice));
        }
    }
    void copyd2h() {
        NVTX_RANGE(__func__);
        if(length > 0) {
            gpuErrchk(cudaMemcpy(h_u32, d_u32, length * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_f, d_f, length * sizeof(float), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_d, d_d, length * sizeof(double), cudaMemcpyDeviceToHost));
        }
    }
};

/**
 * Repeatdly read elements from device data via cuRVE.
 */
__global__ void curveRead(const uint32_t ELEMENT_COUNT, const curve::Curve::VariableHash AGENT_HASH) {
    // Grid stride loop over the problem size
    for (uint32_t idx = (blockDim.x * blockIdx.x) + threadIdx.x; idx < ELEMENT_COUNT; idx += blockDim.x) {
        for(uint32_t multiplier = 0; multiplier < MULTIPLIER; multiplier++) {
            // Read the individuals values of a and b.
            // @todo - getAgnet / getMessage isnt' required outside of flamegpu's curve
            uint32_t id = curve::Curve::getAgentVariable<uint32_t>("u32", AGENT_HASH, idx);
            float x = curve::Curve::getAgentVariable<float>("f", AGENT_HASH, idx);
            double y = curve::Curve::getAgentVariable<double>("d", AGENT_HASH, idx);
        }
    }
}

/**
 * Repeatdly read elements from device data directly.
 */
__global__ void directRead(const uint32_t ELEMENT_COUNT, const uint32_t * const __restrict__ u32s, const float * const __restrict__ fs, const double * const __restrict__ ds) {
    // Grid stride loop over the problem size
    for (uint32_t idx = (blockDim.x * blockIdx.x) + threadIdx.x; idx < ELEMENT_COUNT; idx += blockDim.x) {
        for(uint32_t multiplier = 0; multiplier < MULTIPLIER; multiplier++) {
            // Read the individuals values of a and b.
            // @todo - getAgnet / getMessage isnt' required outside of flamegpu's curve
            uint32_t id = u32s[idx];
            float x = fs[idx];
            double y = ds[idx];
        }
    }
}

/**
 * Repeatdly write elements from device data via cuRVE.
 */
__global__ void curveWrite(const uint32_t ELEMENT_COUNT, const curve::Curve::VariableHash AGENT_HASH) {
    // Grid stride loop over the problem size
    for (uint32_t idx = (blockDim.x * blockIdx.x) + threadIdx.x; idx < ELEMENT_COUNT; idx += blockDim.x) {
        for(uint32_t multiplier = 0; multiplier < MULTIPLIER; multiplier++) {
            curve::Curve::setAgentVariable<uint32_t>("u32", AGENT_HASH, static_cast<uint32_t>(idx), idx);
            curve::Curve::setAgentVariable<float>("f", AGENT_HASH, static_cast<float>(idx), idx);
            curve::Curve::setAgentVariable<double>("d", AGENT_HASH, static_cast<double>(idx), idx);
        }
    }
}

/**
 * Repeatdly write elements from device data directly.
 */
__global__ void directWrite(const uint32_t ELEMENT_COUNT, uint32_t * const u32s, float * const fs, double * const ds) {
    // Grid stride loop over the problem size
    for (uint32_t idx = (blockDim.x * blockIdx.x) + threadIdx.x; idx < ELEMENT_COUNT; idx += blockDim.x) {
        for(uint32_t multiplier = 0; multiplier < MULTIPLIER; multiplier++) {
            u32s[idx] = static_cast<uint32_t>(idx);
            fs[idx] = static_cast<float>(idx);
            ds[idx] = static_cast<double>(idx);
        }
    }
}

/**
 * Set the cuda device, and call the appropraite method to create a cuda context prior to timed regions
 * @param deviceIdx the 0-indexed device index to use
 * @return elapsed time in seconds (via a steady clock timer)
 */
double initialiseGPUContext(int deviceIdx) {
    NVTX_RANGE(__func__);
    // Initialise a cuda context to avoid this being included in the timing.
    util::SteadyClockTimer timer = util::SteadyClockTimer();
    timer.start();
    cudaError_t cudaStatus = cudaSuccess;
    int deviceCount = 0;

    // Get the device count
    cudaStatus = cudaGetDeviceCount(&deviceCount);
    // Error if CUDA error occured
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error: Could not query CUDA device count.\n");
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
    // Error on 0 devices
    if (deviceCount == 0) {
        fprintf(stderr, "Error: no CUDA devices found!\n");
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
    // Error if the requested device is out of range
    if (deviceIdx >= deviceCount) {
        fprintf(stderr, "Error: Invalid CUDA device index %d. %d devices visible.\n", deviceIdx, deviceCount);
        fflush(stderr);
        exit(EXIT_FAILURE);
    } else if (deviceIdx < 0) {
        fprintf(stderr, "Error: CUDA deivce index must be non-negative (%d < 0)\n", deviceIdx);
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
    // Attempt to set the device 
    cudaStatus = cudaSetDevice(deviceIdx);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error occured setting CUDA device to '%d'. (%d available)\n", deviceIdx, deviceCount);
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
    gpuErrchk(cudaFree(nullptr));
    timer.stop();
    double elapsed = timer.getElapsedSeconds();
    return elapsed;
}

/**
 * Get the cudaDeviceProperties struct for a given device by index
 * @param deviceIdx the 0-enumerated device index
 * @return cudaDevicePropertiers struct for the device
 */
cudaDeviceProp getGPUProperties(const int deviceIdx) {
    NVTX_RANGE(__func__);
    cudaDeviceProp props = {};
    gpuErrchk(cudaGetDeviceProperties(&props, deviceIdx));
    return props;
}

/**
 * Get the number of registers for a cuda Kernel
 * @param function
 * @return number of registers required
 */
template <typename F>
int getKernelRegisters(F func) {
    cudaFuncAttributes attribs = {};
    gpuErrchk(cudaFuncGetAttributes(&attribs, func));
    return attribs.numRegs;
}

/**
 * Initialise cuRVE with the namespace, and register all variables required
 * @param ELEMENT_COUNT the number of agents to allow within cuRVE
 * @return elapsed time in seconds
 */
double initialiseCuRVE() {
    NVTX_RANGE(__func__);
    // Get a timer
    std::unique_ptr<util::Timer> timer = getDriverAppropriateTimer();
    // Start recording the time
    timer->start();

    // Initialise curve via purging
    auto &curve = curve::Curve::getInstance(); 
    curve.purge();

    // Report any errors to stdout/stdderr?
    // curveReportErrors(); // @todo - recent curve error reporting

    // Capture the elapsed time and return
    timer->stop();
    double seconds = timer->getElapsedSeconds();
    return seconds;
}

/**
 * Generate initial data using seeded PRNG, for a given agent count.
 * @param SEED the PRNG seed
 * @param data reference to class instance containing host pointers.
 * @return elapsed time in seconds
 */ 
double initialiseData(const uint64_t SEED, HD_DATA &data) {
    NVTX_RANGE(__func__);
    std::unique_ptr<util::Timer> timer = getDriverAppropriateTimer();
    // Start recording the time
    timer->start();

    // Initialise data as pure prng, it doesn't matter.

    std::mt19937_64 prng(SEED);
    std::uniform_real_distribution<float> fdist(0.f, 1.f);
    std::uniform_real_distribution<double> ddist(0., 1.);
    // Initialise data
    for (uint32_t idx = 0u; idx < data.length; idx++)
	{   
        data.h_u32[idx] = idx;
        data.h_f[idx] = fdist(prng);
        data.h_d[idx] = ddist(prng);
		// curveReportErrors(); // @todo - recent curve error reporting
	}

    // Copy data to the device.
    data.copyh2d();

    // Capture the elapsed time and return
    timer->stop();
    double seconds = timer->getElapsedSeconds();
    return seconds;
}

/**
 * Launch the kernel which performs reading via curve
 * @param data reference to class instance containing host pointers.
 * @return the elapsed time in seconds
 */
double launchCurveRead(HD_DATA &data) {
    NVTX_RANGE(__func__);
    // Get a timer
    std::unique_ptr<util::Timer> timer = getDriverAppropriateTimer();
    // Start recording the time
    timer->start();
    if (data.length > 0) {
        // Compute launch bounds via occupancy API
        int blockSize = 0;
        int minGridSize = 0;
        int gridSize = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, curveRead, 0, data.length);
        gridSize = (data.length + blockSize - 1) / blockSize;

        // Map the curve runtime variables
        // This is analgous to flamegpu::CUDAAgent::mapRuntimeVariables and flamegpu::CUDAMessage::mapWriteRuntimeVariables
        auto &curve = curve::Curve::getInstance(); 
        const unsigned int instance_id = 0; 
        const curve::Curve::VariableHash agent_hash = curve::Curve::variableRuntimeHash(AGENT_NAME);
        const curve::Curve::VariableHash func_hash = curve::Curve::variableRuntimeHash("agentOutput");
        const curve::Curve::VariableHash agent_function_hash = agent_hash + func_hash + instance_id;

        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("u32") + agent_function_hash, data.d_u32, sizeof(uint32_t), data.length);
        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("f") + agent_function_hash, data.d_f, sizeof(float), data.length);
        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("d") + agent_function_hash, data.d_d, sizeof(double), data.length);       

        // update the device copy of curve.
        curve.updateDevice();
        // Sync prior to kernel launch if streams are used. If multithreaded single cotnext rdc, const cache curve symbols must be protected while kernels are in flight.
        gpuErrchk(cudaDeviceSynchronize());

        // Launch the kernel
        curveRead<<<gridSize, blockSize>>>(data.length, agent_function_hash);
        gpuErrchkLaunch();

        // Synchronize the device 
        gpuErrchk(cudaDeviceSynchronize());

        // Unregister curve
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("u32") + agent_function_hash);
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("f") + agent_function_hash);
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("d") + agent_function_hash);
    }

    // Report any curve errors
    // curveReportErrors(); // @todo - recent curve error reporting

    // Record the stop event
    timer->stop();
    double elapsedSeconds = timer->getElapsedSeconds();
    return elapsedSeconds;
}

/**
 * Launch the kernel which performs writing via curve
 * @param data reference to class instance containing host pointers.
 * @return the elapsed time in seconds
 */
double launchCurveWrite(HD_DATA &data) {
    NVTX_RANGE(__func__);
    // Get a timer
    std::unique_ptr<util::Timer> timer = getDriverAppropriateTimer();
    // Start recording the time
    timer->start();
    if (data.length > 0) {
        // Compute launch bounds via occupancy API
        int blockSize = 0;
        int minGridSize = 0;
        int gridSize = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, curveWrite, 0, data.length);
        gridSize = (data.length + blockSize - 1) / blockSize;

        // Map the curve runtime variables
        // This is analgous to flamegpu::CUDAAgent::mapRuntimeVariables and flamegpu::CUDAMessage::mapWriteRuntimeVariables
        auto &curve = curve::Curve::getInstance(); 
        const unsigned int instance_id = 0; 
        const curve::Curve::VariableHash agent_hash = curve::Curve::variableRuntimeHash(AGENT_NAME);
        const curve::Curve::VariableHash func_hash = curve::Curve::variableRuntimeHash("agentOutput");
        const curve::Curve::VariableHash agent_function_hash = agent_hash + func_hash + instance_id;

        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("u32") + agent_function_hash, data.d_u32, sizeof(uint32_t), data.length);
        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("f") + agent_function_hash, data.d_f, sizeof(float), data.length);
        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("d") + agent_function_hash, data.d_d, sizeof(double), data.length);       

        // update the device copy of curve.
        curve.updateDevice();
        // Sync prior to kernel launch if streams are used. If multithreaded single cotnext rdc, const cache curve symbols must be protected while kernels are in flight.
        gpuErrchk(cudaDeviceSynchronize());

        // Launch the kernel
        curveWrite<<<gridSize, blockSize>>>(data.length, agent_function_hash);
        gpuErrchkLaunch();

        // Synchronize the device 
        gpuErrchk(cudaDeviceSynchronize());

        // Unregister curve
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("u32") + agent_function_hash);
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("f") + agent_function_hash);
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("d") + agent_function_hash);
    }

    // Report any curve errors
    // curveReportErrors(); // @todo - recent curve error reporting

    // Record the stop event
    timer->stop();
    double elapsedSeconds = timer->getElapsedSeconds();
    return elapsedSeconds;
}

/**
 * Launch the kernel which performs reading via curve
 * @param data reference to class instance containing host pointers.
 * @return the elapsed time in seconds
 */
double launchDirectRead(HD_DATA &data) {
    NVTX_RANGE(__func__);
    // Get a timer
    std::unique_ptr<util::Timer> timer = getDriverAppropriateTimer();
    // Start recording the time
    timer->start();
    if (data.length > 0) {
        // Compute launch bounds via occupancy API
        int blockSize = 0;
        int minGridSize = 0;
        int gridSize = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, directRead, 0, data.length);
        gridSize = (data.length + blockSize - 1) / blockSize;

        // Launch the kernel
        directRead<<<gridSize, blockSize>>>(data.length, data.d_u32, data.d_f, data.d_d);
        gpuErrchkLaunch();

        // Synchronize the device 
        gpuErrchk(cudaDeviceSynchronize());
    }

    // Record the stop event
    timer->stop();
    double elapsedSeconds = timer->getElapsedSeconds();
    return elapsedSeconds;
}

/**
 * Launch the kernel which performs writing via curve
 * @param data reference to class instance containing host pointers.
 * @return the elapsed time in seconds
 */
double launchDirectWrite(HD_DATA &data) {
    NVTX_RANGE(__func__);
    // Get a timer
    std::unique_ptr<util::Timer> timer = getDriverAppropriateTimer();
    // Start recording the time
    timer->start();
    if (data.length > 0) {
        // Compute launch bounds via occupancy API
        int blockSize = 0;
        int minGridSize = 0;
        int gridSize = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, directWrite, 0, data.length);
        gridSize = (data.length + blockSize - 1) / blockSize;

        // Launch the kernel
        directWrite<<<gridSize, blockSize>>>(data.length, data.d_u32, data.d_f, data.d_d);
        gpuErrchkLaunch();

        // Synchronize the device 
        gpuErrchk(cudaDeviceSynchronize());
    }

    // Record the stop event
    timer->stop();
    double elapsedSeconds = timer->getElapsedSeconds();
    return elapsedSeconds;
}


/**
 * Run the mock simulation, recording the durations of the total simulation and each individual time within the simulation
 * @param ITERATIONS the number of mock simulation iterations to run
 * @param ELEMENT_COUNT the number of agents in the mock simulation
 * @param VERBOSITY The degree of verbosity for output, 0 = none, 1 = verbose, 3 = very verbose
 * @return A tuple containing the simulation elapsed time in seconds, mean time per mock agent function and a vector of tuples, containing the per-iteration time and the per mock agent function per iteration.
 */ 
std::tuple<double, std::array<double, 4>, std::vector<std::array<double, 4>>> repeatedKernelRuns(const uint32_t ITERATIONS, const uint32_t ELEMENT_COUNT, const uint32_t VERBOSITY, HD_DATA &data) {
    NVTX_RANGE(__func__);
    // Get a timer
    std::unique_ptr<util::Timer> timer = getDriverAppropriateTimer();
    // Start recording the time
    timer->start();

    std::vector<std::array<double, 4>> perIterationElapsed = {};
    perIterationElapsed.reserve(ITERATIONS);
     
    for (uint32_t iteration = 0; iteration < ITERATIONS; iteration++) {
        NVTX_RANGE("iteration " + std::to_string(iteration));

        double curveWriteSeconds = launchCurveWrite(data);
        double curveReadSeconds = launchCurveRead(data);
        double directWriteSeconds = launchDirectWrite(data);
        double directReadSeconds = launchDirectRead(data);

        // Record the times for later
        perIterationElapsed.push_back({curveWriteSeconds, curveReadSeconds, directWriteSeconds, directReadSeconds});

        if (VERBOSITY > 1) {
            fprintf(stdout, "time: [%u] curveWriteSeconds  %.6f s\n", iteration, curveWriteSeconds);
            fprintf(stdout, "time: [%u] curveReadSeconds   %.6f s\n", iteration, curveReadSeconds);
            fprintf(stdout, "time: [%u] directWriteSeconds %.6f s\n", iteration, directWriteSeconds);
            fprintf(stdout, "time: [%u] directReadSeconds  %.6f s\n", iteration, directReadSeconds);
        }
    }

    // Get the total simulation elapsed time 
    timer->stop();
    double elapsed = timer->getElapsedSeconds();

    std::array<double, 4> meanTimes = {};
    // Get the mean per iteration times.
    double outputSecondsTotal = 0.0;
    for (auto perKernelElapsed : perIterationElapsed) {
        for (uint32_t idx = 0; idx < perKernelElapsed.size(); idx++) {
            meanTimes[idx] += perKernelElapsed[idx];
        }
    }
    for (uint32_t idx = 0; idx < meanTimes.size(); idx++) {
        meanTimes[idx] /= perIterationElapsed.size();
    }
    // Return the timing information
    return {elapsed, meanTimes, perIterationElapsed};
}

/**
 * Struct containing CLI arguments
 * default values are controlled here.
 */
struct CLIArgs {
    uint32_t DEVICE = 0u;
    uint64_t SEED = 0u; 
    uint32_t REPETITIONS = 1u;
    uint32_t ITERATIONS = 8u;
    uint32_t ELEMENT_COUNT = 2u << 13;
    bool FULL_DEVICE = false;
    uint32_t VERBOSITY = false;

    void print(FILE* fp) const {
        if (fp != nullptr) {
            fprintf(fp, "CLIArgs:\n");
            fprintf(fp, "  DEVICE %u\n", DEVICE);
            fprintf(fp, "  SEED %zu\n", SEED);
            fprintf(fp, "  REPETITIONS %u\n", REPETITIONS);
            fprintf(fp, "  ITERATIONS %u\n", ITERATIONS);
            fprintf(fp, "  ELEMENT_COUNT %u\n", ELEMENT_COUNT);
            fprintf(fp, "  FULL_DEVICE %d\n", FULL_DEVICE);
            fprintf(fp, "  VERBOSITY %d\n", VERBOSITY);
            fflush(fp);
        }
    }
};

/**
 * Define and process the CLI
 */
CLIArgs cli(int argc, char * argv[]) {
    NVTX_RANGE(__func__);

    // Declare the struct which will be populated with values 
    CLIArgs args = {};
    
    // Define the CLI11 object
    CLI::App app{"cuRVE ABM-like benchmark"};

    // Define each cli flag, including setting default values.
    app.add_option(
        "-d,--device",
        args.DEVICE,
        "select the GPU to use"
        )->default_val(args.DEVICE);
    app.add_option(
        "-s,--seed",
        args.SEED,
        "PRNG seed"
        )->default_val(args.SEED);
    app.add_option(
        "-r,--repetitions",
        args.REPETITIONS,
        "The number of repetitions for performance averaging"
        )->default_val(args.REPETITIONS);
    app.add_option(
        "-i,--iterations",
        args.ITERATIONS,
        "number of mock simulation iterations"
        )->default_val(args.ITERATIONS);

    // default_val counts as a value, even when not passed so cannot use group / excludes for these. I'm not loving CLI11.
    auto group = app.add_option_group("Scale");
    group->add_option(
        "-c,--element-count",
        args.ELEMENT_COUNT,
        "The number of 'elements' (threads)"
        )->default_val(args.ELEMENT_COUNT);
    group->add_flag(
        "--full-device",
        args.FULL_DEVICE,
        "Use the maximum number of resident threads for the selected GPU"
        );
    // Require 0 or 1 of these options.
    group->require_option(0, 1); 

    app.add_flag(
        "-v,--verbose",
        args.VERBOSITY,
        "Enable verbose output. Repeat for extra verbosity (2 levels?)"
        );
    // app.add_flag("--append", append, "Append to the output file?");
    // app.add_flag("--overwrite", append, "Append to the output file?");
    // CLI::Option *opt = app.add_option("-f,--file,file", file, "File name");

    // Attempt to parse the CLI, exiting if an exception is thrown by CLI11
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        std::exit(app.exit(e));
    }

    // If verbose, print the args
    if (args.VERBOSITY > 0) {
        args.print(stdout);
    }

    // Return the struct
    return args;
}

/**
 * Host main routine, process CLI and launches the appropriate benchmarks.
 * @param argc the number of arguments
 * @param argv the main method argument values
 * @return status code, non zero values imply an error occured
 */
int main(int argc, char * argv[]) {
    NVTX_RANGE(__func__);

    // Process CLI arguments
    const CLIArgs args = cli(argc, argv);

    // Get the CUDA major and minor versions as integers
    constexpr int CUDA_MAJOR = CUDART_VERSION / 1000;
    constexpr int CUDA_MINOR = CUDART_VERSION / 10%100;

    // Select the device, and create a context, returning the time.
    double deviceInitialisationSeconds = initialiseGPUContext(args.DEVICE);

    // Query some device properties
    cudaDeviceProp props = getGPUProperties(args.DEVICE);

    // Compute the maximum resident threads for the selected device, for verbose output / agent population
    const uint32_t maxResidentThreads = props.maxThreadsPerMultiProcessor * props.multiProcessorCount;

    // Get the register use of the kernels for the current device, for verbose/csv output    
    const int curveReadRegs = getKernelRegisters(curveRead);
    const int directReadRegs = getKernelRegisters(directRead);
    const int curveWriteRegs = getKernelRegisters(curveWrite);
    const int directWriteRegs = getKernelRegisters(directWrite);
    
    // Compute the number of threads / agent count to actually use, based on CLI + device properties
    const uint32_t ELEMENT_COUNT = args.FULL_DEVICE ? maxResidentThreads : args.ELEMENT_COUNT;

    // Detect if WDDM is being used, so the most appropraite timer can be used (cudaEvent for linux/tcc, steady clock on windows). Update the anon namespace var.
    deviceUsingWDDM = util::wddm::deviceIsWDDM();

    // Print verbose output about the current benchmark
    if (args.VERBOSITY > 0) {
        fprintf(stdout, "CUDA %d.%d\n", CUDA_MAJOR, CUDA_MINOR);
        fprintf(stdout, "GPU %u: %s, SM_%d%d\n", args.DEVICE, props.name, props.major, props.minor);
        fprintf(stdout, "maxResidentThread: %u\n", maxResidentThreads);
        fprintf(stdout, "ELEMENT_COUNT:       %u (FULL_DEVICE %d)\n", ELEMENT_COUNT, args.FULL_DEVICE);
        fprintf(stdout, "curveWrite  registers: %d\n", curveWriteRegs);
        fprintf(stdout, "curveRead   registers: %d\n", curveReadRegs);
        fprintf(stdout, "directWrite registers: %d\n", directWriteRegs);
        fprintf(stdout, "directRead  registers: %d\n", directReadRegs);
        if (!deviceUsingWDDM) {
            fprintf(stdout, "WDDM=False, using CUDAEventTimer\n");
        } else {
            fprintf(stdout, "WDDM=True,  using SteadyClockTimer\n");
        }
        fprintf(stdout, "time: deviceInitialisation: %.6f s\n", deviceInitialisationSeconds);
    }

    // Prep an object for handling the CSV output, so to de-noise the miniapp
    // @todo not stdout
    util::OutputCSV csv = {stdout};
    csv.setHeader("seed,repetitions,iterations,ELEMENT_COUNT,message_count,message_fraction,CUDA,GPU,ComputeCapability,maxResidentThreads,agentOutputRegisters,agentInputRegisters,deviceInitialisationSeconds,repetition,initialiseCuRVESeconds,initialiseDataSeconds,benchmarkSeconds,outputSecondsMean,inputSecondsMean");

    // Build a row of csv data to print later
    csv.setHeader(std::string("seed,") + 
        "repetitions," + 
        "iterations," + 
        "element_count," + 
        "CUDA," + 
        "GPU," + 
        "ComputeCapability," + 
        "maxResidentThreads," + 
        "curveWriteRegs," + 
        "curveReadRegs," + 
        "directWriteRegs," + 
        "directReadRegs," + 
        "deviceInitialisationSeconds," + 
        "repetition," + 
        "initialiseCuRVESeconds," + 
        "initialiseDataSeconds," + 
        "benchmarkSeconds," + 
        "curveWriteMeanSeconds," + 
        "curveReadMeanSeconds," + 
        "directWriteMeanSeconds," + 
        "directReadMeanSeconds");


    // For up to the right number of repetitions
    for (uint32_t repetition = 0; repetition < args.REPETITIONS; repetition++) {
        NVTX_RANGE("repetition " + repetition);
        // Initialise curve
        double initialiseCuRVESeconds = initialiseCuRVE();

        // Allocate host and device data, this should clean up after itself.
        HD_DATA data(ELEMENT_COUNT);

        // Re-seed the RNG and gnerate initial state
        double initialiseDataSeconds = initialiseData(args.SEED, data);

        // If verbose, output pre-simulation timing info. 
        if (args.VERBOSITY > 0) {
            fprintf(stdout, "time: initialiseCuRVE  %.6f s\n", initialiseCuRVESeconds);
            fprintf(stdout, "time: initialiseData   %.6f s\n", initialiseDataSeconds);
        }

        // Run the mock simulation
        auto [benchmarkSeconds, perKernelMeanSeconds, perIterationElapsed] = repeatedKernelRuns(args.ITERATIONS, ELEMENT_COUNT, args.VERBOSITY, data);

        // Output some crude validation if verbose.
        if (args.VERBOSITY > 0) {
            data.copyd2h();
            constexpr uint32_t VALIDATION_OUTPUT = 4;
            for(unsigned int idx = 0; idx < std::min(VALIDATION_OUTPUT, data.length); idx++) {
                printf("Validation: %u: %u %f %f\n", idx, data.h_u32[idx], data.h_f[idx], data.h_d[idx]);
            }
        }

        // If verbose, output post-simulation timing info
        if (args.VERBOSITY > 0) {
            fprintf(stdout, "time: repeatedKernelRuns %.6f s\n", benchmarkSeconds);
            fprintf(stdout, "time: curveWrite  %.6f s\n", perKernelMeanSeconds[0]);
            fprintf(stdout, "time: curveRead   %.6f s\n", perKernelMeanSeconds[1]);
            fprintf(stdout, "time: directWrite  %.6f s\n", perKernelMeanSeconds[2]);
            fprintf(stdout, "time: directRead %.6f s\n", perKernelMeanSeconds[3]);
        }
    
        // Build a row of csv data to print later
        std::string csvRow = std::to_string(args.SEED) + "," + 
            std::to_string(args.REPETITIONS) + "," + 
            std::to_string(args.ITERATIONS) + "," + 
            std::to_string(ELEMENT_COUNT) + "," + 
            std::to_string(CUDA_MAJOR) + "." + std::to_string(CUDA_MINOR) + "," + 
            props.name + "," + 
            std::to_string((props.major*10)+props.minor) + "," + 
            std::to_string(maxResidentThreads) + "," + 
            std::to_string(curveWriteRegs) + "," + 
            std::to_string(curveReadRegs) + "," + 
            std::to_string(directWriteRegs) + "," + 
            std::to_string(directReadRegs) + "," + 
            std::to_string(deviceInitialisationSeconds) + "," + 
            std::to_string(repetition) + "," + 
            std::to_string(initialiseCuRVESeconds) + "," + 
            std::to_string(initialiseDataSeconds) + "," + 
            std::to_string(benchmarkSeconds) + "," + 
            std::to_string(perKernelMeanSeconds[0]) + "," + 
            std::to_string(perKernelMeanSeconds[1]) + "," + 
            std::to_string(perKernelMeanSeconds[2]) + "," + 
            std::to_string(perKernelMeanSeconds[3]);
        csv.appendRow(csvRow);
    }

    // Write the CSV data out to stdout / disk
    csv.write();

    // Reset the device, to ensure profiling output has completed
    gpuErrchk(cudaDeviceReset());

    // Return a successful code
    return EXIT_SUCCESS;
}
