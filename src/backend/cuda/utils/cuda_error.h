#ifndef CUDA_ERROR_H
#define CUDA_ERROR_H

#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) \
            throw std::runtime_error(std::string("CUDA error: ") \
                + cudaGetErrorString(err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } while(0)

#endif // CUDA_ERROR_H