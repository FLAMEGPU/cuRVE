/**
 * cuRVE benchmark
 * 
 * This contains a becnhmark of cuRVE, mirroring the usage of cuRVE wihin FLAME GPU 2. 
 * It is structured to mirror an agent based model using global communication. 
 * In FLAME GPU 2 terminology, this is mocking a model running for N iterations, with M agents which each read M messages per iteration to perfom some local behaviour.
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

// Env variables as defines for now, for a closer match these would be pulled through curve in their own namespace.
#define G_REPULSE_FACTOR 0.05f
#define G_RADIUS 2.0f
#define AGENT_NAME "circle"
#define MESSAGE_LIST_NAME "message"


/**
 * Class to mock a population of agents, storing agent data SoA-like. This is only used on the host
 */
class MockPopulation {
 public:
    uint32_t length = 0;

    uint32_t * h_id = nullptr;
    float * h_x = nullptr;
    float * h_y = nullptr;
    float * h_z = nullptr;

    float * d_id = nullptr;
    float * d_x = nullptr;
    float * d_y = nullptr;
    float * d_z = nullptr;

    MockPopulation(const uint32_t length) :
        length(0),
        h_id(nullptr),
        h_x(nullptr),
        h_y(nullptr),
        h_z(nullptr),
        d_id(nullptr),
        d_x(nullptr),
        d_y(nullptr),
        d_z(nullptr) {
        this->allocate(length);
    }

    ~MockPopulation() {
        this->deallocate();
    }

    void allocate(const uint32_t length) {
        NVTX_RANGE(__func__);
        if (length != 0) {
            this->deallocate();
        }
        this->length = length;
        this->h_id = static_cast<uint32_t*>(std::malloc(length * sizeof(uint32_t)));
        this->h_x = static_cast<float*>(std::malloc(length * sizeof(float)));
        this->h_y = static_cast<float*>(std::malloc(length * sizeof(float)));
        this->h_z = static_cast<float*>(std::malloc(length * sizeof(float)));
        memset(this->h_id, 0, length * sizeof(uint32_t));
        memset(this->h_x, 0, length * sizeof(float));
        memset(this->h_y, 0, length * sizeof(float));
        memset(this->h_z, 0, length * sizeof(float));

        gpuErrchk(cudaMalloc(&this->d_id, length * sizeof(uint32_t)));
        gpuErrchk(cudaMalloc(&this->d_x, length * sizeof(float)));
        gpuErrchk(cudaMalloc(&this->d_y, length * sizeof(float)));
        gpuErrchk(cudaMalloc(&this->d_z, length * sizeof(float)));

        gpuErrchk(cudaMemset(this->d_id, 0, length * sizeof(uint32_t)));
        gpuErrchk(cudaMemset(this->d_x, 0, length * sizeof(float)));
        gpuErrchk(cudaMemset(this->d_y, 0, length * sizeof(float)));
        gpuErrchk(cudaMemset(this->d_z, 0, length * sizeof(float)));
    };

    void deallocate() {
        NVTX_RANGE(__func__);
        this->length = 0;
        if(this->h_id != nullptr) {
            std::free(this->h_id);
            this->h_id = nullptr;
        }
        if(this->h_x != nullptr) {
            std::free(this->h_x);
            this->h_x = nullptr;
        }
        if(this->h_y != nullptr) {
            std::free(this->h_y);
            this->h_y = nullptr;
        }
        if(this->h_z != nullptr) {
            std::free(this->h_z);
            this->h_z = nullptr;
        }
        if(this->d_x != nullptr) {
            gpuErrchk(cudaFree(this->d_x));
            this->d_x = nullptr;
        }
        if(this->d_y != nullptr) {
            gpuErrchk(cudaFree(this->d_y));
            this->d_y = nullptr;
        }
        if(this->d_z != nullptr) {
            gpuErrchk(cudaFree(this->d_z));
            this->d_z = nullptr;
        }
    };

    void copyh2d() {
        NVTX_RANGE(__func__);
        if(length > 0) {
            gpuErrchk(cudaMemcpy(d_id, h_id, length * sizeof(uint32_t), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_x, h_x, length * sizeof(float), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_y, h_y, length * sizeof(float), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_z, h_z, length * sizeof(float), cudaMemcpyHostToDevice));
        }
    }
    void copyd2h() {
        NVTX_RANGE(__func__);
        if(length > 0) {
            gpuErrchk(cudaMemcpy(h_id, d_id, length * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_x, d_x, length * sizeof(float), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_y, d_y, length * sizeof(float), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_z, d_z, length * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
};

/**
 * Class to mock a message list of agents, storing agent data SoA-like. This is only used on the host
 */
class MockMessageList {
 public:
    uint32_t length = 0;
    uint32_t * d_id = nullptr;
    float * d_x = nullptr;
    float * d_y = nullptr;
    float * d_z = nullptr;

    MockMessageList(const uint32_t length) :
        length(0),
        d_id(nullptr),
        d_x(nullptr),
        d_y(nullptr),
        d_z(nullptr) {
        this->allocate(length);
    }

    ~MockMessageList() {
        this->deallocate();
    }

    void allocate(const uint32_t length) {
        NVTX_RANGE(__func__);
        if (length != 0) {
            this->deallocate();
        }
        this->length = length;
        gpuErrchk(cudaMalloc(&this->d_id, length * sizeof(uint32_t)));
        gpuErrchk(cudaMalloc(&this->d_x, length * sizeof(float)));
        gpuErrchk(cudaMalloc(&this->d_y, length * sizeof(float)));
        gpuErrchk(cudaMalloc(&this->d_z, length * sizeof(float)));
        gpuErrchk(cudaMemset(this->d_x, 0, length * sizeof(uint32_t)));
        gpuErrchk(cudaMemset(this->d_x, 0, length * sizeof(float)));
        gpuErrchk(cudaMemset(this->d_y, 0, length * sizeof(float)));
        gpuErrchk(cudaMemset(this->d_z, 0, length * sizeof(float)));
    };

    void deallocate() {
        NVTX_RANGE(__func__);
        this->length = 0;
        if(this->d_id != nullptr) {
            gpuErrchk(cudaFree(this->d_id));
            this->d_id = nullptr;
        }
        if(this->d_x != nullptr) {
            gpuErrchk(cudaFree(this->d_x));
            this->d_x = nullptr;
        }
        if(this->d_y != nullptr) {
            gpuErrchk(cudaFree(this->d_y));
            this->d_y = nullptr;
        }
        if(this->d_z != nullptr) {
            gpuErrchk(cudaFree(this->d_z));
            this->d_z = nullptr;
        }
    };
};


/**
 * Mock agent message output function
 * Each individual writes out it's personal data to global memory via curve
 */
__global__ void agentOutput(const uint32_t AGENT_COUNT, const curve::Curve::VariableHash AGENT_HASH, const curve::Curve::VariableHash MESSAGE_HASH) {
    // Grid stride loop over the problem size
    for (uint32_t idx = (blockDim.x * blockIdx.x) + threadIdx.x; idx < AGENT_COUNT; idx += blockDim.x) {
        // Read the individuals values of a and b.
        // @todo - getAgnet / getMessage isnt' required outside of flamegpu's curve
        uint32_t id = curve::Curve::getAgentVariable<float>("id", AGENT_HASH, idx);
        float x = curve::Curve::getAgentVariable<float>("x", AGENT_HASH, idx);
        float y = curve::Curve::getAgentVariable<float>("y", AGENT_HASH, idx);
        float z = curve::Curve::getAgentVariable<float>("z", AGENT_HASH, idx);
        // Write them out to the message locations
        curve::Curve::setMessageVariable<float>("id", MESSAGE_HASH, id, idx);
        curve::Curve::setMessageVariable<float>("x", MESSAGE_HASH, x, idx);
        curve::Curve::setMessageVariable<float>("y", MESSAGE_HASH, y, idx);
        curve::Curve::setMessageVariable<float>("z", MESSAGE_HASH, z, idx);
        // Report curve errors from the device. This has been replaced in FLAME GPU 2's curve?
        // curveReportLastDeviceError(); // @todo - replace error checking
    }
}

/**
 * Mock agent message input/iteration function
 * Each individual reads data in from each message, does some operations with it, then writes it out to global memory via curve
 */
__global__ void agentInput(const uint32_t AGENT_COUNT, const uint32_t MESSAGE_COUNT, const curve::Curve::VariableHash AGENT_HASH, const curve::Curve::VariableHash MESSAGE_HASH) {
    // Grid stride loop over the problem size
    for (uint32_t idx = (blockDim.x * blockIdx.x) + threadIdx.x; idx < AGENT_COUNT; idx += blockDim.x) {
        uint32_t agent_id = curve::Curve::getAgentVariable<float>("id", AGENT_HASH, idx);
        const float REPULSE_FACTOR = G_REPULSE_FACTOR; // @todo store in curve environment namespace
        const float RADIUS = G_RADIUS; // @todo store in curve environment namespace
        float fx = 0.0;
        float fy = 0.0;
        float fz = 0.0;
        const float x1 = curve::Curve::getAgentVariable<float>("x", AGENT_HASH, idx);
        const float y1 = curve::Curve::getAgentVariable<float>("y", AGENT_HASH, idx);
        const float z1 = curve::Curve::getAgentVariable<float>("z", AGENT_HASH, idx);
        int count = 0;
        // Mock message iteration loop
        for (uint32_t messageIdx = 0; messageIdx < MESSAGE_COUNT; messageIdx++) {
            const uint32_t message_id = curve::Curve::getMessageVariable<float>("z", MESSAGE_HASH, idx);
            if (message_id != agent_id) {
                const float x2 = curve::Curve::getMessageVariable<float>("id", MESSAGE_HASH, idx);
                const float y2 = curve::Curve::getMessageVariable<float>("x", MESSAGE_HASH, idx);
                const float z2 = curve::Curve::getMessageVariable<float>("y", MESSAGE_HASH, idx);
                float x21 = x2 - x1;
                float y21 = y2 - y1;
                float z21 = z2 - z1;
                const float separation = sqrtf(x21*x21 + y21*y21 + z21*z21);
                if (separation < RADIUS && separation > 0.0f) {
                    float k = sinf((separation / RADIUS) * 3.141f * -2) * REPULSE_FACTOR;
                    // Normalise without recalculating separation
                    x21 /= separation;
                    y21 /= separation;
                    z21 /= separation;
                    fx += k * x21;
                    fy += k * y21;
                    fz += k * z21;
                    count++;
                }
            }
        }
        fx /= count > 0 ? count : 1;
        fy /= count > 0 ? count : 1;
        fz /= count > 0 ? count : 1;
        // Write to the agent's individual data
        curve::Curve::setAgentVariable<float>("x", AGENT_HASH, x1 + fx, idx);
        curve::Curve::setAgentVariable<float>("y", AGENT_HASH, y1 + fy, idx);
        curve::Curve::setAgentVariable<float>("z", AGENT_HASH, z1 + fz, idx);
        // curve::Curve::setAgentVariable<float>("drift", AGENT_HASH, sqrtf(fx*fx + fy*fy + fz*fz), idx);
        // Report curve errors from the device. This has been replaced in FLAME GPU 2's curve?
        // curveReportLastDeviceError();
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
 * @param population reference to class instance containing host pointers.
 * @return elapsed time in seconds
 */ 
double initialiseData(const uint64_t SEED, MockPopulation &population) {
    NVTX_RANGE(__func__);
    std::unique_ptr<util::Timer> timer = getDriverAppropriateTimer();
    // Start recording the time
    timer->start();

    // Compute the environment width from the agent population
    const float ENV_MAX = static_cast<float>(floor(cbrtf(population.length)));

    std::mt19937_64 prng(SEED);
    std::uniform_real_distribution<float> dist(0.f, ENV_MAX);
    // Initialise data
    for (uint32_t idx = 0u; idx < population.length; idx++)
	{   
        population.h_x[idx] = idx;
        population.h_x[idx] = dist(prng);
        population.h_y[idx] = dist(prng);
        population.h_z[idx] = dist(prng);
		// curveReportErrors(); // @todo - recent curve error reporting
	}

    // Copy data to the device.
    population.copyh2d();

    // Capture the elapsed time and return
    timer->stop();
    double seconds = timer->getElapsedSeconds();
    return seconds;
}

/**
 * Launch the kernel which mocks up agent message output
 * @param population reference to class instance containing host pointers.
 * @param messageList reference to class instance containing host pointers.
 * @return the elapsed time in seconds
 */
double launchAgentOutput(MockPopulation &population, MockMessageList &messageList) {
    NVTX_RANGE(__func__);
    // Get a timer
    std::unique_ptr<util::Timer> timer = getDriverAppropriateTimer();
    // Start recording the time
    timer->start();
    if (population.length > 0) {
        // Compute launch bounds via occupancy API
        int blockSize = 0;
        int minGridSize = 0;
        int gridSize = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, agentOutput, 0, population.length);
        gridSize = (population.length + blockSize - 1) / blockSize;

        // Map the curve runtime variables
        // This is analgous to flamegpu::CUDAAgent::mapRuntimeVariables and flamegpu::CUDAMessage::mapWriteRuntimeVariables
        auto &curve = curve::Curve::getInstance(); 
        const unsigned int instance_id = 0; 
        const curve::Curve::VariableHash agent_hash = curve::Curve::variableRuntimeHash(AGENT_NAME);
        const curve::Curve::VariableHash func_hash = curve::Curve::variableRuntimeHash("agentOutput");
        const curve::Curve::VariableHash agent_function_hash = agent_hash + func_hash + instance_id;
        const curve::Curve::VariableHash message_hash = curve.variableRuntimeHash(MESSAGE_LIST_NAME);
        const curve::Curve::VariableHash agent_function_message_hash = agent_hash + func_hash + message_hash + instance_id;

        // const curve::Curve::VariableHash message_function_hash = agent_hash + func_hash + instance_id; // @todo

        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("id") + agent_function_hash, population.d_id, sizeof(uint32_t), population.length);
        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("x") + agent_function_hash, population.d_x, sizeof(float), population.length);
        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("y") + agent_function_hash, population.d_y, sizeof(float), population.length);
        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("z") + agent_function_hash, population.d_z, sizeof(float), population.length);

        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("id") + agent_function_message_hash, messageList.d_id, sizeof(uint32_t), messageList.length);
        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("x") + agent_function_message_hash, messageList.d_x, sizeof(float), messageList.length);
        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("y") + agent_function_message_hash, messageList.d_y, sizeof(float), messageList.length);
        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("z") + agent_function_message_hash, messageList.d_z, sizeof(float), messageList.length);

        // update the device copy of curve.
        curve.updateDevice();
        // Sync prior to kernel launch if streams are used. If multithreaded single cotnext rdc, const cache curve symbols must be protected while kernels are in flight.
        gpuErrchk(cudaDeviceSynchronize());

        // Launch the kernel
        agentOutput<<<gridSize, blockSize>>>(population.length, agent_function_hash, agent_function_message_hash);
        gpuErrchkLaunch();

        // Synchronize the device 
        gpuErrchk(cudaDeviceSynchronize());

        // Unregister curve
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("id") + agent_function_hash);
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("x") + agent_function_hash);
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("y") + agent_function_hash);
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("z") + agent_function_hash);

        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("id") + agent_function_message_hash);
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("x") + agent_function_message_hash);
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("y") + agent_function_message_hash);
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("z") + agent_function_message_hash);
    }

    // Report any curve errors
    // curveReportErrors(); // @todo - recent curve error reporting

    // Record the stop event
    timer->stop();
    double elapsedSeconds = timer->getElapsedSeconds();
    return elapsedSeconds;
}

/**
 * Launch the kernel which mocks up brute force agent message input 
 * @param MESSAGE_COUNT the number of messages (if population sizes grow too large, to improve runtime without reducing occupancy)
 * @param population reference to class instance containing host pointers.
 * @param messageList reference to class instance containing host pointers.
 * @return the elapsed time in seconds
 */
double launchAgentInput(const uint32_t MESSAGE_COUNT, MockPopulation &population, MockMessageList &messageList) {
    NVTX_RANGE(__func__);
    // Get a timer
    std::unique_ptr<util::Timer> timer = getDriverAppropriateTimer();
    // Start recording the time
    timer->start();

    if (population.length > 0) {
        // Compute launch bounds via occupancy API
        int blockSize = 0;
        int minGridSize = 0;
        int gridSize = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, agentInput, 0, population.length);
        gridSize = (population.length + blockSize - 1) / blockSize;

        // Map the curve runtime variables
        // This is analgous to flamegpu::CUDAAgent::mapRuntimeVariables
        auto &curve = curve::Curve::getInstance(); 
        const unsigned int instance_id = 0; 
        const curve::Curve::VariableHash agent_hash = curve::Curve::variableRuntimeHash(AGENT_NAME);
        const curve::Curve::VariableHash func_hash = curve::Curve::variableRuntimeHash("agentInput");
        const curve::Curve::VariableHash agent_function_hash = agent_hash + func_hash + instance_id;
        const curve::Curve::VariableHash message_hash = curve.variableRuntimeHash(MESSAGE_LIST_NAME);
        const curve::Curve::VariableHash agent_function_message_hash = agent_hash + func_hash + message_hash + instance_id;


        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("id") + agent_function_hash, population.d_id, sizeof(uint32_t), population.length);
        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("x") + agent_function_hash, population.d_x, sizeof(float), population.length);
        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("y") + agent_function_hash, population.d_y, sizeof(float), population.length);
        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("z") + agent_function_hash, population.d_z, sizeof(float), population.length);

        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("id") + agent_function_message_hash, messageList.d_id, sizeof(uint32_t), messageList.length);
        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("x") + agent_function_message_hash, messageList.d_x, sizeof(float), messageList.length);
        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("y") + agent_function_message_hash, messageList.d_y, sizeof(float), messageList.length);
        curve.registerVariableByHash(curve::Curve::variableRuntimeHash("z") + agent_function_message_hash, messageList.d_z, sizeof(float), messageList.length);


        // update the device copy of curve.
        curve.updateDevice();
        // Sync prior to kernel launch if streams are used. If multithreaded single cotnext rdc, const cache curve symbols must be protected while kernels are in flight.
        gpuErrchk(cudaDeviceSynchronize());

        // Launch the kernel
        agentInput<<<gridSize, blockSize>>>(population.length, MESSAGE_COUNT, agent_function_hash, agent_function_message_hash);
        gpuErrchkLaunch();

        // Synchronize the device 
        gpuErrchk(cudaDeviceSynchronize());

        // Unnmap curve
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("id") + agent_function_hash);
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("x") + agent_function_hash);
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("y") + agent_function_hash);
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("z") + agent_function_hash);

        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("id") + agent_function_message_hash);
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("x") + agent_function_message_hash);
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("y") + agent_function_message_hash);
        curve.unregisterVariableByHash(curve::Curve::variableRuntimeHash("z") + agent_function_message_hash);
    }
    // Report any curve errors
    // curveReportErrors(); // @todo - recent curve error reporting

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
std::tuple<double, double, double, std::vector<std::tuple<double, double>>> mockSimulation(const uint32_t ITERATIONS, const uint32_t AGENT_COUNT, const uint32_t MESSAGE_COUNT, const uint32_t VERBOSITY, MockPopulation &population, MockMessageList &messageList) {
    NVTX_RANGE(__func__);
    // Get a timer
    std::unique_ptr<util::Timer> timer = getDriverAppropriateTimer();
    // Start recording the time
    timer->start();

    std::vector<std::tuple<double, double>> perIterationElapsed = {};
    perIterationElapsed.reserve(ITERATIONS);
     
    for (uint32_t iteration = 0; iteration < ITERATIONS; iteration++) {
        NVTX_RANGE("iteration " + std::to_string(iteration));
        // Run the mock message output function
        double outputSeconds = launchAgentOutput(population, messageList);

        // Run the mock message input function
        double inputSeconds = launchAgentInput(MESSAGE_COUNT, population, messageList);

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
    uint32_t REPETITIONS = 1u;
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
        double initialiseCuRVESeconds = initialiseCuRVE();

        // Allocate host and device data, this should clean up after itself.
        MockPopulation population(AGENT_COUNT);
        MockMessageList messageList(AGENT_COUNT);

        // Re-seed the RNG and gnerate initial state
        double initialiseDataSeconds = initialiseData(args.SEED, population);

        // If verbose, output pre-simulation timing info. 
        if (args.VERBOSITY > 0) {
            fprintf(stdout, "time: initialiseCuRVE  %.6f s\n", initialiseCuRVESeconds);
            fprintf(stdout, "time: initialiseData   %.6f s\n", initialiseDataSeconds);
        }

        // Run the mock simulation
        auto [mockSimulationSeconds, outputSecondsMean, inputSecondsMean, perIterationElapsed] = mockSimulation(args.ITERATIONS, AGENT_COUNT, MESSAGE_COUNT, args.VERBOSITY, population, messageList);

        // Output some crude validation if verbose.
        if (args.VERBOSITY > 0) {
            population.copyd2h();
            constexpr uint32_t VALIDATION_OUTPUT = 4;
            for(unsigned int idx = 0; idx < std::min(VALIDATION_OUTPUT, population.length); idx++) {
                printf("Validation: %u: %f %f %f\n", idx, population.h_x[idx], population.h_y[idx], population.h_z[idx]);
            }
        }

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
