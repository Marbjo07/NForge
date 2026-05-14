#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "nforge/core/tensor_layout.h"

__device__ __forceinline__ size_t physicalOffsetCUDA(size_t linear, const TensorLayout& L);

__global__ void addKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout, 
    const float* __restrict__ rhs, const TensorLayout rhsLayout, 
    float* __restrict__ out, const TensorLayout outLayout, size_t count);

__global__ void subKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout, 
    const float* __restrict__ rhs, const TensorLayout rhsLayout, 
    float* __restrict__ out, const TensorLayout outLayout, size_t count);

__global__ void mulKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout, 
    const float* __restrict__ rhs, const TensorLayout rhsLayout, 
    float* __restrict__ out, const TensorLayout outLayout, size_t count);

__global__ void divKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout, 
    const float* __restrict__ rhs, const TensorLayout rhsLayout, 
    float* __restrict__ out, const TensorLayout outLayout, size_t count);


__global__ void fillKernel(float* __restrict__ data, float value, size_t count);

__global__ void setKernel(
    float* __restrict__ dst, const TensorLayout dstLayout,
    const float* __restrict__ src, const TensorLayout srcLayout, size_t count);

__global__ void checkAllEqualKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout,
    const float* __restrict__ rhs, const TensorLayout rhsLayout,
    int* isEqualFlag, size_t count);

__global__ void iaddKernel(float* __restrict__ lhs, const TensorLayout lhsLayout, 
    const float* __restrict__ rhs, const TensorLayout rhsLayout, size_t count);

__global__ void isubKernel(float* __restrict__ lhs, const TensorLayout lhsLayout, 
    const float* __restrict__ rhs, const TensorLayout rhsLayout, size_t count);

__global__ void imulKernel(float* __restrict__ lhs, const TensorLayout lhsLayout, 
    const float* __restrict__ rhs, const TensorLayout rhsLayout, size_t count);
    
__global__ void idivKernel(float* __restrict__ lhs, const TensorLayout lhsLayout, 
    const float* __restrict__ rhs, const TensorLayout rhsLayout, size_t count);

#endif // KERNELS_CUH