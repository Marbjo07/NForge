#ifndef CUDA_CONTEXT_H
#define CUDA_CONTEXT_H


#include "cuda_error.h"


class CudaContext {
public:
    static CudaContext& get() {
        static CudaContext instance;
        return instance;
    }
    
    cudaStream_t stream() const { 
        return m_stream; 
    }

    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;
    CudaContext(CudaContext&&) = delete;
    CudaContext& operator=(CudaContext&&) = delete;

private:
    cudaStream_t m_stream;

    CudaContext() {
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaFree(0));
        CUDA_CHECK(cudaStreamCreate(&m_stream));
        CUDA_CHECK(cudaGetLastError());
    }

    ~CudaContext() {
        cudaStreamDestroy(m_stream);
    }
};

#endif // CUDA_CONTEXT_H