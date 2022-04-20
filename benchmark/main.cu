/**
 * cuRVE benchmark
 * 
 * This contains a becnhmark of cuRVE, mirroring the usage of cuRVE wihin FLAME GPU 2. 
 * It is structured to mirror an agent based model using global communication. 
 * In FLAME GPU 2 terminology, this is mocking a model running for N iterations, with M agents which each read M messages per iteration to perfom some local behaviour.
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
#include "curve.cuh"

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

// Define constant expression string literals for commonly used variables to pass around.
// I would much rather these be constexpr strings, but the relaxed constexpr flag wasn't cooperating
#define AGENT_A "a"
#define AGENT_B "b"
#define AGENT_C "c"
// @todo - more recent curve writes messages to a different namespace? Need to check this difference.
#define MESSAGE_A "m_a"
#define MESSAGE_B "m_b"

/**
 * Mock agent message output function
 * Each individual writes out it's personal data to global memory via curve
 */
__global__ void agentOutput(const uint32_t AGENT_COUNT) {
    // Grid stride loop over the problem size
    for (uint32_t idx = (blockDim.x * blockIdx.x) + threadIdx.x; idx < AGENT_COUNT; idx += blockDim.x) {
        // Read the individuals values of a and b.
        float a = getFloatVariable(AGENT_A, idx);
        float b = getFloatVariable(AGENT_B, idx);
        // Write them out to the message locations
        setFloatVariable(MESSAGE_A, a, idx);
        setFloatVariable(MESSAGE_B, b, idx);
        // Report curve errors from the device. This has been replaced in FLAME GPU 2's curve?
        curveReportLastDeviceError();
    }
}

/**
 * Mock agent message input/iteration function
 * Each individual reads data in from each message, does some operations with it, then writes it out to global memory via curve
 */
__global__ void agentInput(const uint32_t AGENT_COUNT, const uint32_t MESSAGE_COUNT) {
    // Grid stride loop over the problem size
    for (uint32_t idx = (blockDim.x * blockIdx.x) + threadIdx.x; idx < AGENT_COUNT; idx += blockDim.x) {
        // thread local value for accumulation
        float c = 0.0;
        // Mock message iteration loop
        for (uint32_t messageIdx = 0; messageIdx < MESSAGE_COUNT; messageIdx++) {
            // Read values from messages
            float message_a = getFloatVariable(MESSAGE_A, messageIdx);
            float message_b = getFloatVariable(MESSAGE_B, messageIdx);
            // Accumulate into the c value
            c += (message_a + message_b);
        }
        // Normalise by the number of messages
        c /= MESSAGE_COUNT;
        // Write to the agent's individual data
        setFloatVariable(AGENT_C, c, idx);
        // Report curve errors from the device. This has been replaced in FLAME GPU 2's curve?
        curveReportLastDeviceError();
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
 * Get the cudaFuncAttributes structs for the mock agent functions as a tuple
 * @return tuple of 2 cudaFuncAttributes [agentOutputAttribtues, agentInputAttributes]
 */
std::tuple<cudaFuncAttributes, cudaFuncAttributes> getKernelAttributes() {
    cudaFuncAttributes agentOutputAttributes = {};
    cudaFuncAttributes agentInputAttributes = {};

    gpuErrchk(cudaFuncGetAttributes(&agentOutputAttributes, agentOutput));
    gpuErrchk(cudaFuncGetAttributes(&agentInputAttributes, agentInput));

    return {agentOutputAttributes, agentInputAttributes};
}


/**
 * Initialise cuRVE with the namespace, and register all variables required
 * @param AGENT_COUNT the number of agents to allow within cuRVE
 * @return elapsed time in seconds
 */
double initialiseCuRVE(const uint32_t AGENT_COUNT) {
    NVTX_RANGE(__func__);
    // Get a timer
    std::unique_ptr<util::Timer> timer = getDriverAppropriateTimer();
    // Start recording the time
    timer->start();

    // Initialise curve
    curveInit(AGENT_COUNT);
    // Set the curve namespace
    curveSetNamespace("name1");
    // Registar all variables with curve
    curveRegisterVariable(AGENT_A);
    curveRegisterVariable(AGENT_B);
    curveRegisterVariable(AGENT_C);
    curveRegisterVariable(MESSAGE_A);
    curveRegisterVariable(MESSAGE_B);
    // Report any errors to stdout/stdderr?
    curveReportErrors();

    // Capture the elapsed time and return
    timer->stop();
    double seconds = timer->getElapsedSeconds();
    return seconds;
}

/**
 * Generate initial data using seeded PRNG, for a given agent count.
 * @param SEED the PRNG seed
 * @param AGENT_COUNT the number agents to initialise data for
 * @return elapsed time in seconds
 */ 
double initialiseData(const uint64_t SEED, const uint32_t AGENT_COUNT) {
    NVTX_RANGE(__func__);
    std::unique_ptr<util::Timer> timer = getDriverAppropriateTimer();
    // Start recording the time
    timer->start();

    std::mt19937_64 prng(SEED);
    std::uniform_real_distribution<float> a_dist(0.f, 1.f);
    // Initialise data
    for (uint32_t idx = 0u; idx < AGENT_COUNT; idx++)
	{   
    	float a = a_dist(prng);
		float b = static_cast<float>(idx);
    	curveSetFloat(AGENT_A, a, idx);
		curveSetFloat(AGENT_B, b, idx);
		curveSetFloat(AGENT_C, 0, idx);
		curveSetFloat(MESSAGE_A, 0, idx);
		curveSetFloat(MESSAGE_B, 0, idx);
		curveReportErrors();
	}

    // Capture the elapsed time and return
    timer->stop();
    double seconds = timer->getElapsedSeconds();
    return seconds;
}

/**
 * Launch the kernel which mocks up agent message output
 * @param AGENT_COUNT the number of agents (i.e. active threads)
 * @return the elapsed time in seconds
 */
double launchAgentOutput(const uint32_t AGENT_COUNT) {
    NVTX_RANGE(__func__);
    // Get a timer
    std::unique_ptr<util::Timer> timer = getDriverAppropriateTimer();
    // Start recording the time
    timer->start();

    // Compute launch bounds via occupancy API
    int blockSize = 0;
    int minGridSize = 0;
    int gridSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, agentOutput, 0, AGENT_COUNT);
    gridSize = (AGENT_COUNT + blockSize - 1) / blockSize;

    // Launch the kernel
    agentOutput<<<gridSize, blockSize>>>(AGENT_COUNT);
    gpuErrchkLaunch();

    // Synchronize the device 
    gpuErrchk(cudaDeviceSynchronize());

    // Report any curve errors
    curveReportErrors();

    // Record the stop event
    timer->stop();
    double elapsedSeconds = timer->getElapsedSeconds();
    return elapsedSeconds;
}

/**
 * Launch the kernel which mocks up brute force agent message input 
 * @param AGENT_COUNT the number of agents (i.e. active threads)
 * @param MESSAGE_COUNT the number of messages (if population sizes grow too large, to improve runtime without reducing occupancy)
 * @return the elapsed time in seconds
 */
double launchAgentInput(const uint32_t AGENT_COUNT, const uint32_t MESSAGE_COUNT) {
    NVTX_RANGE(__func__);
    // Get a timer
    std::unique_ptr<util::Timer> timer = getDriverAppropriateTimer();
    // Start recording the time
    timer->start();

    // Compute launch bounds via occupancy API
    int blockSize = 0;
    int minGridSize = 0;
    int gridSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, agentOutput, 0, AGENT_COUNT);
    gridSize = (AGENT_COUNT + blockSize - 1) / blockSize;

    // Launch the kernel
    agentInput<<<gridSize, blockSize>>>(AGENT_COUNT, MESSAGE_COUNT);
    gpuErrchkLaunch();

    // Synchronize the device 
    gpuErrchk(cudaDeviceSynchronize());

    // Report any curve errors
    curveReportErrors();

    // Record the stop event
    timer->stop();
    double elapsedSeconds = timer->getElapsedSeconds();
    return elapsedSeconds;
}

/**
 * Run the mock simulation, recording the durations of the total simulation and each individual time within the simulation
 * @param ITERATIONS the number of mock simulation iterations to run
 * @param AGENT_COUNT the number of agents in the mock simulation
 * @param MESSAGE_COUNT the number of messages to consider within the mock simulation
 * @param VERBOSITY The degree of verbosity for output, 0 = none, 1 = verbose, 3 = very verbose
 * @return A tuple containing the simulation elapsed time in seconds, mean time per mock agent function and a vector of tuples, containing the per-iteration time and the per mock agent function per iteration.
 */ 
std::tuple<double, double, double, std::vector<std::tuple<double, double>>> mockSimulation(const uint32_t ITERATIONS, const uint32_t AGENT_COUNT, const uint32_t MESSAGE_COUNT, const uint32_t VERBOSITY) {
    NVTX_RANGE("__func__");
    // Get a timer
    std::unique_ptr<util::Timer> timer = getDriverAppropriateTimer();
    // Start recording the time
    timer->start();

    std::vector<std::tuple<double, double>> perIterationElapsed = {};
    perIterationElapsed.reserve(ITERATIONS);
     
    for (uint32_t iteration = 0; iteration < ITERATIONS; iteration++) {
        NVTX_RANGE("iteration " + std::to_string(iteration));
        // Run the mock message output function
        double outputSeconds = launchAgentOutput(AGENT_COUNT);

        // Run the mock message input function
        double inputSeconds = launchAgentInput(AGENT_COUNT, MESSAGE_COUNT);

        // Record the times for later
        perIterationElapsed.push_back({outputSeconds, inputSeconds});

        if (VERBOSITY > 1) {
            fprintf(stdout, "time: [%u] agentOutput  %.6f s\n", iteration, outputSeconds);
            fprintf(stdout, "time: [%u] agentInput   %.6f s\n", iteration, inputSeconds);
        }
    }

    // Get the total simulation elapsed time 
    timer->stop();
    double simulationElapsed = timer->getElapsedSeconds();

    // Get the mean per iteration times.
    double outputSecondsTotal = 0.0;
    double inputSecondsTotal = 0.0;
    for (auto [outputSeconds, inputSeconds] : perIterationElapsed) {
        outputSecondsTotal += outputSeconds;
        inputSecondsTotal += inputSeconds;
    }
    double outputSecondsMean = outputSecondsTotal / perIterationElapsed.size();
    double inputSecondsMean = inputSecondsTotal / perIterationElapsed.size();
    // Return the timing information
    return {simulationElapsed, outputSecondsMean, inputSecondsMean, perIterationElapsed};
}

/**
 * Struct containing CLI arguments
 * default values are controlled here.
 */
struct CLIArgs {
    uint32_t DEVICE = 0u;
    uint64_t SEED = 0u; 
    uint32_t REPETITIONS = 3u;
    uint32_t ITERATIONS = 8u;
    uint32_t AGENT_COUNT = 2u << 13;
    float MESSAGE_FRACTION = 1.f;
    bool FULL_DEVICE = false;
    uint32_t VERBOSITY = false;

    void print(FILE* fp) const {
        if (fp != nullptr) {
            fprintf(fp, "CLIArgs:\n");
            fprintf(fp, "  DEVICE %u\n", DEVICE);
            fprintf(fp, "  SEED %zu\n", SEED);
            fprintf(fp, "  REPETITIONS %u\n", REPETITIONS);
            fprintf(fp, "  ITERATIONS %u\n", ITERATIONS);
            fprintf(fp, "  AGENT_COUNT %u\n", AGENT_COUNT);
            fprintf(fp, "  MESSAGE_FRACTION %.3f\n", MESSAGE_FRACTION);
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
    app.add_option(
        "-m,--message-fraction",
        args.MESSAGE_FRACTION,
        "The fraction of 'messages' read by each 'agent'"
        )->default_val(args.MESSAGE_FRACTION)->expected(0.0,1.0);

    // default_val counts as a value, even when not passed so cannot use group / excludes for these. I'm not loving CLI11.
    auto group = app.add_option_group("Scale");
    group->add_option(
        "-c,--agent-count",
        args.AGENT_COUNT,
        "The number of 'agents' (threads)"
        )->default_val(args.AGENT_COUNT);
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

    // Get the register use of the two kernels for the current device, for verbose/csv output
    auto [agentOutputAttribtues, agentInputAttributes] = getKernelAttributes();

    // Compute the number of threads / agent count to actually use, based on CLI + device properties
    const uint32_t AGENT_COUNT = args.FULL_DEVICE ? maxResidentThreads : args.AGENT_COUNT;
    // Compute the number of messages to read
    const uint32_t MESSAGE_COUNT = std::min(static_cast<uint32_t>(std::round(AGENT_COUNT * args.MESSAGE_FRACTION)), AGENT_COUNT);

    // Detect if WDDM is being used, so the most appropraite timer can be used (cudaEvent for linux/tcc, steady clock on windows). Update the anon namespace var.
    deviceUsingWDDM = util::wddm::deviceIsWDDM();

    // Print verbose output about the current benchmark
    if (args.VERBOSITY > 0) {
        fprintf(stdout, "CUDA %d.%d\n", CUDA_MAJOR, CUDA_MINOR);
        fprintf(stdout, "GPU %u: %s, SM_%d%d\n", args.DEVICE, props.name, props.major, props.minor);
        fprintf(stdout, "maxResidentThread: %u\n", maxResidentThreads);
        fprintf(stdout, "AGENT_COUNT:       %u (FULL_DEVICE %d)\n", AGENT_COUNT, args.FULL_DEVICE);
        fprintf(stdout, "MESSAGE_COUNT:     %u\n", MESSAGE_COUNT);
        fprintf(stdout, "agentOutput registers: %d\n", agentOutputAttribtues.numRegs);
        fprintf(stdout, "agentInput  registers: %d\n", agentInputAttributes.numRegs);
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
    csv.setHeader("seed,repetitions,iterations,agent_count,message_count,message_fraction,CUDA,GPU,ComputeCapability,maxResidentThreads,agentOutputRegisters,agentInputRegisters,deviceInitialisationSeconds,repetition,initialiseCuRVESeconds,initialiseDataSeconds,mockSimulationSeconds,outputSecondsMean,inputSecondsMean");


    // For up to the right number of repetitions
    for (uint32_t repetition = 0; repetition < args.REPETITIONS; repetition++) {
        NVTX_RANGE("repetition " + repetition);
        // Initialise curve
        double initialiseCuRVESeconds = initialiseCuRVE(AGENT_COUNT);

        // Re-seed the RNG and gnerate initial state
        double initialiseDataSeconds = initialiseData(args.SEED, AGENT_COUNT);

        // If verbose, output pre-simulation timing info. 
        if (args.VERBOSITY > 0) {
            fprintf(stdout, "time: initialiseCuRVE  %.6f s\n", initialiseCuRVESeconds);
            fprintf(stdout, "time: initialiseData   %.6f s\n", initialiseDataSeconds);
        }

        // Run the mock simulation
        auto [mockSimulationSeconds, outputSecondsMean, inputSecondsMean, perIterationElapsed] = mockSimulation(args.ITERATIONS, AGENT_COUNT, MESSAGE_COUNT, args.VERBOSITY);

        // If verbose, output post-simulation timing info
        if (args.VERBOSITY > 0) {
            fprintf(stdout, "time: mockSimulation   %.6f s\n", mockSimulationSeconds);
            fprintf(stdout, "time: agentOutputMean  %.6f s\n", outputSecondsMean);
            fprintf(stdout, "time: agentInputMean   %.6f s\n", inputSecondsMean);
        }
    
        // Build a row of csv data to print later
        std::string csvRow = std::to_string(args.SEED) + "," + 
            std::to_string(args.REPETITIONS) + "," + 
            std::to_string(args.ITERATIONS) + "," + 
            std::to_string(AGENT_COUNT) + "," + 
            std::to_string(MESSAGE_COUNT) + "," + 
            std::to_string(args.MESSAGE_FRACTION) + "," + 
            std::to_string(CUDA_MAJOR) + "." + std::to_string(CUDA_MINOR) + "," + 
            props.name + "," + 
            std::to_string((props.major*10)+props.minor) + "," + 
            std::to_string(maxResidentThreads) + "," + 
            std::to_string(agentOutputAttribtues.numRegs) + "," + 
            std::to_string(agentInputAttributes.numRegs) + "," + 
            std::to_string(deviceInitialisationSeconds) + "," + 
            std::to_string(repetition) + "," + 
            std::to_string(initialiseCuRVESeconds) + "," + 
            std::to_string(initialiseDataSeconds) + "," + 
            std::to_string(mockSimulationSeconds) + "," + 
            std::to_string(outputSecondsMean) + "," + 
            std::to_string(inputSecondsMean);
        csv.appendRow(csvRow);
    }

    // Write the CSV data out to stdout / disk
    csv.write();

    // Reset the device, to ensure profiling output has completed
    gpuErrchk(cudaDeviceReset());

    // Return a successful code
    return EXIT_SUCCESS;
}
