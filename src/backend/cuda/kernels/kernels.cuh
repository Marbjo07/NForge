#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "nforge/core/tensor_layout.h"

__device__ __forceinline__ int physicalOffsetCUDA(int linear, const TensorLayout& L);

__global__ void addKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout, 
    const float* __restrict__ rhs, const TensorLayout rhsLayout, 
    float* __restrict__ out, const TensorLayout outLayout, int count);

__global__ void subKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout, 
    const float* __restrict__ rhs, const TensorLayout rhsLayout, 
    float* __restrict__ out, const TensorLayout outLayout, int count);

__global__ void mulKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout, 
    const float* __restrict__ rhs, const TensorLayout rhsLayout, 
    float* __restrict__ out, const TensorLayout outLayout, int count);

__global__ void divKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout, 
    const float* __restrict__ rhs, const TensorLayout rhsLayout, 
    float* __restrict__ out, const TensorLayout outLayout, int count);


__global__ void fillKernel(float* __restrict__ data, float value, int count);

__global__ void setKernel(
    float* __restrict__ dst, const TensorLayout dstLayout,
    const float* __restrict__ src, const TensorLayout srcLayout, int count);

__global__ void checkAllEqualKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout,
    const float* __restrict__ rhs, const TensorLayout rhsLayout,
    int* isEqualFlag, int count);

#endif // KERNELS_CUH