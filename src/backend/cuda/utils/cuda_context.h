#ifndef CUDA_CONTEXT_H
#define CUDA_CONTEXT_H

#include "cuda_error.h"

#include <curand.h>

class CudaContext {
public:
    static CudaContext& get() {
        static CudaContext instance;
        return instance;
    }
    
    cudaStream_t stream() const { 
        return m_stream; 
    }

    curandGenerator_t& rng() { return m_gen; }

    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;
    CudaContext(CudaContext&&) = delete;
    CudaContext& operator=(CudaContext&&) = delete;

private:
    cudaStream_t m_stream;
    curandGenerator_t m_gen;

    CudaContext() {
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaFree(0));

        CUDA_CHECK(cudaStreamCreate(&m_stream));

        CURAND_CHECK(curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetStream(m_gen, m_stream));

        CUDA_CHECK(cudaGetLastError());
    }

    ~CudaContext() {
        curandDestroyGenerator(m_gen);

        cudaStreamDestroy(m_stream);
    }
};

#endif // CUDA_CONTEXT_H