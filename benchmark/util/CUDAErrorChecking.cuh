#ifndef INCLUDE_UTIL_CUDAERRORCHECKING_CUH_
#define INCLUDE_UTIL_CUDAERRORCHECKING_CUH_

#include <cuda_runtime.h>
#include <cstdio>
#include <string>
// #include <stdexcept>

namespace util {

/**
 * Error check function for safe CUDA API calling
 * Wrap any cuda runtime API calls with this macro to automatically check the returned cudaError_t
 */
#define gpuErrchk(ans) { util::gpuAssert((ans), __FILE__, __LINE__); }
/**
 * Error check function for safe CUDA API calling
 * @param code CUDA Runtime API return code
 * @param file File where errorcode was reported (e.g. __FILE__)
 * @param line Line no where errorcode was reported (e.g. __LINE__)
 * @throws CUDAError If code != cudaSuccess
 */
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        // THROW exception::CUDAError("CUDA Error: %s(%d): %s", file, line, cudaGetErrorString(code));
        fprintf(stderr, "CUDA Error: %s(%d): %s\n", file, line, cudaGetErrorString(code));
    }
}

/**
 * Error check function for safe CUDA Driver API calling
 * Wrap any cuda drive API calls with this macro to automatically check the returned CUresult
 */
#define gpuErrchkDriverAPI(ans) { util::gpuAssert((ans), __FILE__, __LINE__); }

/**
 * Error check function for use after asynchronous methods
 * Call this macro function after async calls such as kernel launches to automatically check the latest error
 * In debug builds this will perform a synchronisation to catch any errors, in non-debug builds errors may be propagated.
 */
#define gpuErrchkLaunch() { util::gpuLaunchAssert(__FILE__, __LINE__); }
 /**
  * Error check function for checking for the most recent error
  * @param file File where errorcode was reported (e.g. __FILE__)
  * @param line Line no where errorcode was reported (e.g. __LINE__)
  * @throws CUDAError If code != cudaSuccess
  * @see gpuAssert(cudaError_t code, const char *file, int line)
  * @note Only synchronises in debug builds
  */
inline void gpuLaunchAssert(const char *file, int line) {
#ifdef _DEBUG
    gpuAssert(cudaDeviceSynchronize(), file, line);
#endif
    gpuAssert(cudaPeekAtLastError(), file, line);
}

}  // namespace util

#endif  // INCLUDE_UTIL_CUDAERRORCHECKING_CUH_
