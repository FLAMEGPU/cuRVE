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

#include "cuda_runtime.h"

// Include the curve header
#include "curve.h"

// Include a number of utilty classes to simplify code in the example itself.
#include "util/Timer.h"
#include "util/SteadyClockTimer.h"
#include "util/CUDAErrorChecking.cuh"
#include "util/CUDAEventTimer.cuh"
#include "util/wddm.cuh"
#include "util/nvtx.h"

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
    NVTX_RANGE("initialiseGPUContext");
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
    if (deviceIdx > deviceCount) {
        fprintf(stderr, "Error: device %d > %d\n", deviceIdx, deviceCount);
        fflush(stderr);
        exit(EXIT_FAILURE);
    } else if (deviceIdx < 0) {
        fprintf(stderr, "Error: device %d is invalid\n", deviceIdx);
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
    // Attempt to set the device 
    cudaStatus = cudaSetDevice(deviceIdx);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error occured setting CUDA device to '%d'. (%d available)", deviceIdx, deviceCount);
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
    NVTX_RANGE("initialiseCuRVE");
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
    NVTX_RANGE("initialiseData");
    std::unique_ptr<util::Timer> timer = getDriverAppropriateTimer();
    // Start recording the time
    timer->start();

    // @todo - use modern PRNG
    // @todo seed the prng.

    // Initialise data
    for (uint32_t idx = 0u; idx < AGENT_COUNT; idx++)
	{   
    	float a = rand()/(float)RAND_MAX;
		float b = (float) idx;
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
    NVTX_RANGE("launchAgentOutput");
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
    NVTX_RANGE("launchAgentInput");
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
 * Run the mock simulation, recording the durations of the total simulation and each individual time within the simulation
 * @param ITERATIONS the number of mock simulation iterations to run
 * @param AGENT_COUNT the number of agents in the mock simulation
 * @param MESSAGE_COUNT the number of messages to consider within the mock simulation
 * @return A tuple containing the simulation elapsed time in seconds, and a vector of tuples, containing the per-iteration time and the per mock agent function per iteration.
 */ 
std::tuple<double, std::vector<std::tuple<double, double>>> mockSimulation(const uint32_t ITERATIONS, const uint32_t AGENT_COUNT, const uint32_t MESSAGE_COUNT) {
    NVTX_RANGE("mockSimulation");
    // Get a timer
    std::unique_ptr<util::Timer> timer = getDriverAppropriateTimer();
    // Start recording the time
    timer->start();

    std::vector<std::tuple<double, double>> perIterationElapsed = {};
    perIterationElapsed.reserve(ITERATIONS);
     
    for (uint32_t iteration = 0; iteration < ITERATIONS; iteration++) {
        NVTX_RANGE("iteration");  // @todo - add the iteration number to the range name?

        // Run the mock message output function
        double outputSeconds = launchAgentOutput(AGENT_COUNT);

        // Run the mock message input function
        double inputSeconds = launchAgentInput(AGENT_COUNT, MESSAGE_COUNT);

        // Record the times for later
        perIterationElapsed.push_back({outputSeconds, inputSeconds});
    }

    // Get the total simulation elapsed time 
    timer->stop();
    double simulationElapsed = timer->getElapsedSeconds();
    // Return the timing information
    return {simulationElapsed, perIterationElapsed};
}

/**
 * Output the header row of a csv file containing the 
 * @param fp the filepointer to the file to write to.
 */
void outputCSVHeader(FILE * fp) {
    if (fp != nullptr) {
        fprintf(fp, "@todo\n");
    }
}

/**
 * Output the body row of a csv file, containing data about a specific mock simulation
 * @param fp the filepointer to the file to write to.
 */
void outputCSVRow(FILE * fp) {
    if (fp != nullptr) {
        fprintf(fp, "@todo\n");
    }
}


/**
 * Host main routine, process CLI and launches the appropriate benchmarks.
 * @param argc the number of arguments
 * @param argv the main method argument values
 * @return status code, non zero values imply an error occured
 */
int main(int argc, char * argv[]) {
    NVTX_RANGE("main");

    // @todo - arg parsing from command line / configuraiton file.
    // @todo - multiple agent count / message counts?
    const uint32_t REPETITIONS = 3u;
    // const uint32_t AGENT_COUNT = 2048u;
    // const uint32_t MESSAGE_COUNT = AGENT_COUNT;
    const uint64_t SEED = 12u;
    const uint32_t DEVICE = 0u;
    const uint32_t ITERATIONS = 8u;
    const bool APPEND_FLAG = false;
    const bool VERBOSE = true;


    // Select the device, and create a context, returning the time.
    double deviceInitialisationSeconds = initialiseGPUContext(DEVICE);
    if (VERBOSE) {
        printf("deviceInitialisationSeconds %.6f s\n", deviceInitialisationSeconds);
    }
    
    // Get device properties
    cudaDeviceProp props = getGPUProperties(DEVICE);
    const int maxResidentThreads = props.maxThreadsPerMultiProcessor * props.multiProcessorCount;
    if (VERBOSE) {
        printf("GPU %d: %s, SM_%d%d, CUDA %d.%d, maxResidentThreads %d\n", DEVICE, props.name, props.major, props.minor, CUDART_VERSION/1000, CUDART_VERSION/10%100, maxResidentThreads);
    }

    const uint32_t AGENT_COUNT = maxResidentThreads;
    const uint32_t MESSAGE_COUNT = AGENT_COUNT;

    // Get the register use for each function
    auto [agentOutputAttribtues, agentInputAttributes] = getKernelAttributes();
    if (VERBOSE) {
        printf("Register use: agentOutput %d, agentInput %d\n", agentOutputAttribtues.numRegs, agentInputAttributes.numRegs);

    }


    // Detect if WDDM is being used, so the most appropraite timer can be used (cudaEvent for linux/tcc, steady clock on windows). Update the anon namespace var.
    deviceUsingWDDM = util::wddm::deviceIsWDDM();
    if (VERBOSE) {
        if (deviceUsingWDDM) {
            printf("Device is not WDDM, using CUDAEventTimer\n");
        } else {
            printf("Device is WDDM, using SteadyClockTimer\n");
        }
    }

    // For up to the right number of repetitions
    for (uint32_t repetition = 0; repetition < REPETITIONS; repetition++) {
        // Initialise curve
        double initialiseCuRVESeconds = initialiseCuRVE(AGENT_COUNT);
        if (VERBOSE) {
            printf("initialiseCuRVE %.6f s\n", deviceInitialisationSeconds);
        }

        // Re-seed the RNG and gnerate initial state
        double initialiseDataSeconds = initialiseData(SEED, AGENT_COUNT);
        if (VERBOSE) {
            printf("initialiseData %.6f s\n", initialiseDataSeconds);
        }

        // Run the mock simulation
        auto [simulationElapsed, perIterationElapsed] = mockSimulation(ITERATIONS, AGENT_COUNT, MESSAGE_COUNT);
        if (VERBOSE) {
            printf("mockSimulation %.6f s\n", simulationElapsed);
            for (auto [outputSeconds, intputSeconds] : perIterationElapsed) {
                printf("outputSeconds %.6f s\n", outputSeconds);
                printf("intputSeconds %.6f s\n", intputSeconds);
            }
        }
        
        // Output data
        printf("@todo - output csv\n");

    }

    // Reset the device, to ensure profiling output has completed
    gpuErrchk(cudaDeviceReset());
    return EXIT_SUCCESS;
}
