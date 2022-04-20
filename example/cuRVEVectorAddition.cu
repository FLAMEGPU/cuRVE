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

// For the CUDA runtime routines (prefixed with "cuda_")
#include "cuda_runtime.h"
#include "curve/curve.cuh"




__global__ void vectorAdd(int numElements)
{
    unsigned int idx;
	float a, b, c;

    idx = (blockDim.x * blockIdx.x) + threadIdx.x;

    a = getFloatVariable("a", idx);
    b = getFloatVariable("b", idx);
    //printf("B is %f \n", b);

    c = a + b;

    setFloatVariable("c", c, idx);

    curveReportLastDeviceError();
}

/**
 * Host main routine
 */
int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    int numElements = 256;

    curveInit(numElements);
    curveSetNamespace("name1");
    curveRegisterVariable("a");
    curveRegisterVariable("b");
    curveRegisterVariable("c");
    curveReportErrors();

    printf("[Vector addition of %d elements]\n", numElements);


    for (int i = 0; i < numElements; ++i)
	{
    	float a = rand()/(float)RAND_MAX;
		float b = (float) i;
    	curveSetFloat("a", a, i);
		curveSetFloat("b", b, i);
		curveReportErrors();
	}

    //timing start
    cudaEvent_t start,stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);


    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(numElements);
    err = cudaGetLastError();

    //timing end
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);

    printf("Kernel Time was %f ms\n", elapsedTime);

    curveReportErrors();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
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
