/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <random>

// For the CUDA runtime routines (prefixed with "cuda_")
#include "cuda_runtime.h"
#include "curve/curve.cuh"




__global__ void vectorAdd(int NUM_ELEMENTS, curve::Curve::VariableHash namespace_hash)
{
    unsigned int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    if(idx < NUM_ELEMENTS){
        // @todo - getAgnet / getMessage isnt' required outside of flamegpu's curve
        float a = curve::Curve::getAgentVariable<float>("a", namespace_hash, idx);
        float b = curve::Curve::getAgentVariable<float>("b", namespace_hash, idx);
        float c = a + b;

        // if (idx < 32) {
        //     printf("%d: %f %f %f\n", idx, a, b, c);
        // }

        // curve::Curve::setAgentVariable<float>("c", namespace_hash, c, idx);

        // @todo - no seatbelts / device errors setup yet.
        // curveReportLastDeviceError();
    }
}

/**
 * Host main routine
 */
int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    const int NUM_ELEMENTS = 1024;
    const unsigned int SEED = 12;

    // Summon curve singleton.
    auto &curve = curve::Curve::getInstance(); 

    // Initialise curve via purging.
    curve.purge();

    // Allocate hsot memory for insiitalstion
    float * h_a = static_cast<float*>(malloc(NUM_ELEMENTS * sizeof(float)));
    float * h_b = static_cast<float*>(malloc(NUM_ELEMENTS * sizeof(float)));
    float * h_c = static_cast<float*>(malloc(NUM_ELEMENTS * sizeof(float)));

    // Allocate device memory for data to be stored in.
    float * d_a = nullptr;
    float * d_b = nullptr;
    float * d_c = nullptr;
    cudaMalloc(&d_a, NUM_ELEMENTS * sizeof(float));
    cudaMalloc(&d_b, NUM_ELEMENTS * sizeof(float));
    cudaMalloc(&d_c, NUM_ELEMENTS * sizeof(float));



    // Populate host data
    std::mt19937_64 prng(SEED);
    std::uniform_real_distribution<float> a_dist(0.f, 1.f);
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
    	h_a[i] = a_dist(prng);
		h_b[i] = (float) i;
        h_c[i] = 0;
	}
    // Copy to the device
    cudaMemcpy(d_a, h_a, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);


    printf("[Vector addition of %d elements]\n", NUM_ELEMENTS);


    // Register variables with curve in an appropraite namesapce, in real world use this needs doing any time that the device pointer may have changed.
    // This is analgous to flamegpu::CUDAAgent::mapRuntimeVariables
    // @todo - not all of these are required in this example.
    const unsigned int instance_id = 0; 
    const curve::Curve::VariableHash agent_hash = curve::Curve::variableRuntimeHash("agent");
    const curve::Curve::VariableHash func_hash = curve::Curve::variableRuntimeHash("func");
    const curve::Curve::VariableHash agent_function_hash = agent_hash + func_hash + instance_id;


    curve.registerVariableByHash(curve::Curve::variableRuntimeHash("a") + agent_function_hash, d_a, sizeof(float), NUM_ELEMENTS);
    curve.registerVariableByHash(curve::Curve::variableRuntimeHash("b") + agent_function_hash, d_b, sizeof(float), NUM_ELEMENTS);
    curve.registerVariableByHash(curve::Curve::variableRuntimeHash("c") + agent_function_hash, d_c, sizeof(float), NUM_ELEMENTS);
    // update the device copy of curve.
    curve.updateDevice();
    // Sync prior to kernel launch if streams are used. If multithreaded single cotnext rdc, const cache curve symbols must be protected while kernels are in flight.
    cudaDeviceSynchronize();
    // curveReportErrors(); // @todo - update?


    //timing start
    cudaEvent_t start,stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);


    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(NUM_ELEMENTS + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(NUM_ELEMENTS, agent_function_hash);
    err = cudaGetLastError();
    // Sync the device
    cudaDeviceSynchronize();

    //timing end
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);

    printf("Kernel Time was %f ms\n", elapsedTime);

    // curveReportErrors();

    // Unregister curve use, analagous to flamegpu::   CUDAAgent::unmapRuntimeVariables

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free memory
    if (h_a != nullptr) {
        std::free(h_a);
        h_a = nullptr;
    }
    if (h_b != nullptr) {
        std::free(h_b);
        h_b = nullptr;
    }
    if (h_c != nullptr) {
        std::free(h_c);
        h_c = nullptr;
    }
    if (d_a != nullptr) {
        cudaFree(d_a);
        d_a = nullptr;
    }
    if (d_b != nullptr) {
        cudaFree(d_b);
        d_b = nullptr;
    }
    if (d_c != nullptr) {
        cudaFree(d_c);
        d_c = nullptr;
    }
    // Reset the device and exit
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    

    printf("Done\n");
    return 0;
}
