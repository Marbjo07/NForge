#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "backend/cuda/utils/cuda_utils.h"
#include "nforge/core/tensor_layout.h"

__device__ __forceinline__ size_t physicalOffsetCUDA(size_t linear, const TensorLayout& L);

// binary operations
__global__ void addKernel(const float* __restrict__ lhs, const TensorLayout lhsLayout,
                          const float* __restrict__ rhs, const TensorLayout rhsLayout,
                          float* __restrict__ out, const TensorLayout outLayout, size_t count);

__global__ void subKernel(const float* __restrict__ lhs, const TensorLayout lhsLayout,
                          const float* __restrict__ rhs, const TensorLayout rhsLayout,
                          float* __restrict__ out, const TensorLayout outLayout, size_t count);

__global__ void mulKernel(const float* __restrict__ lhs, const TensorLayout lhsLayout,
                          const float* __restrict__ rhs, const TensorLayout rhsLayout,
                          float* __restrict__ out, const TensorLayout outLayout, size_t count);

__global__ void divKernel(const float* __restrict__ lhs, const TensorLayout lhsLayout,
                          const float* __restrict__ rhs, const TensorLayout rhsLayout,
                          float* __restrict__ out, const TensorLayout outLayout, size_t count);

__global__ void fillKernel(float* __restrict__ data, float value, size_t count);

__global__ void setKernel(float* __restrict__ dst, const TensorLayout dstLayout,
                          const float* __restrict__ src, const TensorLayout srcLayout,
                          size_t count);

__global__ void checkAllEqualKernel(const float* __restrict__ lhs, const TensorLayout lhsLayout,
                                    const float* __restrict__ rhs, const TensorLayout rhsLayout,
                                    int* isEqualFlag, size_t count);


__global__ void equalKernel(const float* __restrict__ lhs, const TensorLayout lhsLayout,
                            const float* __restrict__ rhs, const TensorLayout rhsLayout,
                            float* __restrict__ out, const TensorLayout outLayout, size_t count);

__global__ void notEqualKernel(const float* __restrict__ lhs, const TensorLayout lhsLayout,
                               const float* __restrict__ rhs, const TensorLayout rhsLayout,
                               float* __restrict__ out, const TensorLayout outLayout, size_t count);

__global__ void lessKernel(const float* __restrict__ lhs, const TensorLayout lhsLayout,
                           const float* __restrict__ rhs, const TensorLayout rhsLayout,
                           float* __restrict__ out, const TensorLayout outLayout, size_t count);

__global__ void lessEqualKernel(const float* __restrict__ lhs, const TensorLayout lhsLayout,
                                const float* __restrict__ rhs, const TensorLayout rhsLayout,
                                float* __restrict__ out, const TensorLayout outLayout,
                                size_t count);

__global__ void greaterKernel(const float* __restrict__ lhs, const TensorLayout lhsLayout,
                              const float* __restrict__ rhs, const TensorLayout rhsLayout,
                              float* __restrict__ out, const TensorLayout outLayout, size_t count);

__global__ void greaterEqualKernel(const float* __restrict__ lhs, const TensorLayout lhsLayout,
                                   const float* __restrict__ rhs, const TensorLayout rhsLayout,
                                   float* __restrict__ out, const TensorLayout outLayout,
                                   size_t count);

__global__ void isCloseKernel(const float* __restrict__ lhs, const TensorLayout lhsLayout,
                              const float* __restrict__ rhs, const TensorLayout rhsLayout,
                              float* __restrict__ out, const TensorLayout outLayout, size_t count,
                              float tolerance);

// in-place kernels
__global__ void iaddKernel(float* __restrict__ lhs, const TensorLayout lhsLayout,
                           const float* __restrict__ rhs, const TensorLayout rhsLayout,
                           size_t count);

__global__ void isubKernel(float* __restrict__ lhs, const TensorLayout lhsLayout,
                           const float* __restrict__ rhs, const TensorLayout rhsLayout,
                           size_t count);

__global__ void imulKernel(float* __restrict__ lhs, const TensorLayout lhsLayout,
                           const float* __restrict__ rhs, const TensorLayout rhsLayout,
                           size_t count);

__global__ void idivKernel(float* __restrict__ lhs, const TensorLayout lhsLayout,
                           const float* __restrict__ rhs, const TensorLayout rhsLayout,
                           size_t count);

__global__ void isqrtKernel(float* __restrict__ data, size_t count);

// reduction kernels
__global__ void squareSumKernel(const float* __restrict__ data, float* result, size_t count);

__global__ void sumReductionKernel(const float* __restrict__ data, float* result,
                                   const TensorLayout layout, size_t blockCount,
                                   const TensorLayout outLayout, size_t outCount);

__global__ void minReductionKernel(const float* __restrict__ data, float* result,
                                   const TensorLayout layout, size_t blockCount,
                                   const TensorLayout outLayout, size_t outCount);

__global__ void maxReductionKernel(const float* __restrict__ data, float* result,
                                   const TensorLayout layout, size_t blockCount,
                                   const TensorLayout outLayout, size_t outCount);

__global__ void prodReductionKernel(const float* __restrict__ data, float* result,
                                    const TensorLayout layout, size_t blockCount,
                                    const TensorLayout outLayout, size_t outCount);

__global__ void allReductionKernel(const float* __restrict__ data, float* result,
                                   const TensorLayout layout, size_t blockCount,
                                   const TensorLayout outLayout, size_t outCount);

__global__ void anyReductionKernel(const float* __restrict__ data, float* result,
                                   const TensorLayout layout, size_t blockCount,
                                   const TensorLayout outLayout, size_t outCount);


__global__ void matmulKernel(const float* __restrict__ lhs, const TensorLayout lhsLayout,
                             const float* __restrict__ rhs, const TensorLayout rhsLayout,
                             float* __restrict__ out, const TensorLayout outLayout, size_t batch,
                             size_t m, size_t k, size_t p);

#endif  // KERNELS_CUH