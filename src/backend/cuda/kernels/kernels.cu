#include "backend/cuda/kernels/kernels.cuh"

__device__ __forceinline__ int physicalOffsetCUDA(int linear, const TensorLayout& L) {
    int off = L.offset;
    for (int d = L.rank - 1; d >= 0; d--) {
        int coord = linear % L.shape[d];
        linear /= L.shape[d];
        off += coord * L.strides[d];
    }
    return off;
}

__global__ void addKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout, 
    const float* __restrict__ rhs, const TensorLayout rhsLayout, 
    float* __restrict__ out, const TensorLayout outLayout, int count) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count) return;

    int lhsIdx = physicalOffsetCUDA(i, lhsLayout);
    int rhsIdx = physicalOffsetCUDA(i, rhsLayout);
    int outIdx = physicalOffsetCUDA(i, outLayout);

    out[outIdx] = lhs[lhsIdx] + rhs[rhsIdx];
}

__global__ void subKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout,
    const float* __restrict__ rhs, const TensorLayout rhsLayout,
    float* __restrict__ out, const TensorLayout outLayout, int count) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count) return;

    int lhsIdx = physicalOffsetCUDA(i, lhsLayout);
    int rhsIdx = physicalOffsetCUDA(i, rhsLayout);
    int outIdx = physicalOffsetCUDA(i, outLayout);

    out[outIdx] = lhs[lhsIdx] - rhs[rhsIdx];
}

__global__ void mulKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout,
    const float* __restrict__ rhs, const TensorLayout rhsLayout,
    float* __restrict__ out, const TensorLayout outLayout, int count) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count) return;

    int lhsIdx = physicalOffsetCUDA(i, lhsLayout);
    int rhsIdx = physicalOffsetCUDA(i, rhsLayout);
    int outIdx = physicalOffsetCUDA(i, outLayout);

    out[outIdx] = lhs[lhsIdx] * rhs[rhsIdx];
}

__global__ void divKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout,
    const float* __restrict__ rhs, const TensorLayout rhsLayout,
    float* __restrict__ out, const TensorLayout outLayout, int count) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count) return;

    int lhsIdx = physicalOffsetCUDA(i, lhsLayout);
    int rhsIdx = physicalOffsetCUDA(i, rhsLayout);
    int outIdx = physicalOffsetCUDA(i, outLayout);

    out[outIdx] = lhs[lhsIdx] / rhs[rhsIdx];
}


__global__ void fillKernel(float* __restrict__ data, float value, int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) data[i] = value;
}


__global__ void setKernel(
    float* __restrict__ dst, const TensorLayout dstLayout,
    const float* __restrict__ src, const TensorLayout srcLayout, int count) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int dstIdx = physicalOffsetCUDA(i, dstLayout);
    int srcIdx = physicalOffsetCUDA(i, srcLayout);
    dst[dstIdx] = src[srcIdx];
}

__global__ void checkAllEqualKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout,
    const float* __restrict__ rhs, const TensorLayout rhsLayout,
    int* isEqualFlag, int count) {

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int lhsIdx = physicalOffsetCUDA(i, lhsLayout);
    int rhsIdx = physicalOffsetCUDA(i, rhsLayout);
    if (lhs[lhsIdx] != rhs[rhsIdx]) *isEqualFlag = 0;
}