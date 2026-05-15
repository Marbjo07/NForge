#include "backend/cuda/kernels/kernels.cuh"

__device__ __forceinline__ size_t physicalOffsetCUDA(size_t linear, const TensorLayout& L) {
    size_t off = L.offset;
    for (int d = (int)L.rank - 1; d >= 0; d--) {
        size_t coord = linear % L.shape[d];
        linear /= L.shape[d];
        off += coord * L.strides[d];
    }
    return off;
}

__global__ void addKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout,
    const float* __restrict__ rhs, const TensorLayout rhsLayout,
    float* __restrict__ out, const TensorLayout outLayout, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count) return;

    size_t lhsIdx = physicalOffsetCUDA(i, lhsLayout);
    size_t rhsIdx = physicalOffsetCUDA(i, rhsLayout);
    size_t outIdx = physicalOffsetCUDA(i, outLayout);

    out[outIdx] = lhs[lhsIdx] + rhs[rhsIdx];
}

__global__ void subKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout,
    const float* __restrict__ rhs, const TensorLayout rhsLayout,
    float* __restrict__ out, const TensorLayout outLayout, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count) return;

    size_t lhsIdx = physicalOffsetCUDA(i, lhsLayout);
    size_t rhsIdx = physicalOffsetCUDA(i, rhsLayout);
    size_t outIdx = physicalOffsetCUDA(i, outLayout);

    out[outIdx] = lhs[lhsIdx] - rhs[rhsIdx];
}

__global__ void mulKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout,
    const float* __restrict__ rhs, const TensorLayout rhsLayout,
    float* __restrict__ out, const TensorLayout outLayout, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count) return;

    size_t lhsIdx = physicalOffsetCUDA(i, lhsLayout);
    size_t rhsIdx = physicalOffsetCUDA(i, rhsLayout);
    size_t outIdx = physicalOffsetCUDA(i, outLayout);

    out[outIdx] = lhs[lhsIdx] * rhs[rhsIdx];
}

__global__ void divKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout,
    const float* __restrict__ rhs, const TensorLayout rhsLayout,
    float* __restrict__ out, const TensorLayout outLayout, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count) return;

    size_t lhsIdx = physicalOffsetCUDA(i, lhsLayout);
    size_t rhsIdx = physicalOffsetCUDA(i, rhsLayout);
    size_t outIdx = physicalOffsetCUDA(i, outLayout);

    out[outIdx] = lhs[lhsIdx] / rhs[rhsIdx];
}

__global__ void fillKernel(float* __restrict__ data, float value, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) data[i] = value;
}

__global__ void setKernel(
    float* __restrict__ dst, const TensorLayout dstLayout,
    const float* __restrict__ src, const TensorLayout srcLayout, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    size_t dstIdx = physicalOffsetCUDA(i, dstLayout);
    size_t srcIdx = physicalOffsetCUDA(i, srcLayout);
    dst[dstIdx] = src[srcIdx];
}

__global__ void checkAllEqualKernel(
    const float* __restrict__ lhs, const TensorLayout lhsLayout,
    const float* __restrict__ rhs, const TensorLayout rhsLayout,
    int* isEqualFlag, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    size_t lhsIdx = physicalOffsetCUDA(i, lhsLayout);
    size_t rhsIdx = physicalOffsetCUDA(i, rhsLayout);
    if (lhs[lhsIdx] != rhs[rhsIdx]) *isEqualFlag = 0;
}

__global__ void iaddKernel(float* __restrict__ lhs, const TensorLayout lhsLayout,
                           const float* __restrict__ rhs, const TensorLayout rhsLayout, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count) return;

    size_t lhsIdx = physicalOffsetCUDA(i, lhsLayout);
    size_t rhsIdx = physicalOffsetCUDA(i, rhsLayout);
    lhs[lhsIdx] += rhs[rhsIdx];
}

__global__ void isubKernel(float* __restrict__ lhs, const TensorLayout lhsLayout,
                           const float* __restrict__ rhs, const TensorLayout rhsLayout, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count) return;

    size_t lhsIdx = physicalOffsetCUDA(i, lhsLayout);
    size_t rhsIdx = physicalOffsetCUDA(i, rhsLayout);
    lhs[lhsIdx] -= rhs[rhsIdx];
}

__global__ void imulKernel(float* __restrict__ lhs, const TensorLayout lhsLayout,
                           const float* __restrict__ rhs, const TensorLayout rhsLayout, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count) return;

    size_t lhsIdx = physicalOffsetCUDA(i, lhsLayout);
    size_t rhsIdx = physicalOffsetCUDA(i, rhsLayout);
    lhs[lhsIdx] *= rhs[rhsIdx];
}

__global__ void idivKernel(float* __restrict__ lhs, const TensorLayout lhsLayout,
                           const float* __restrict__ rhs, const TensorLayout rhsLayout, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count) return;

    size_t lhsIdx = physicalOffsetCUDA(i, lhsLayout);
    size_t rhsIdx = physicalOffsetCUDA(i, rhsLayout);
    lhs[lhsIdx] /= rhs[rhsIdx];
}

__global__ void isqrtKernel(float* __restrict__ data, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count) return;
    data[i] = sqrt(data[i]);
}

__global__ void squareSumKernel(const float* __restrict__ data, float* result, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count) return;
    float x = data[i] * data[i];

    atomicAdd(result, x);
}

__device__ static float atomicMin(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        // Use fminf to find the max of the current value and the new value
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);  // Repeat if another thread updated the memory
    return __int_as_float(old);
}

__device__ static float atomicMax(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        // Use fmaxf to find the max of the current value and the new value
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);  // Repeat if another thread updated the memory
    return __int_as_float(old);
}

__device__ static float atomicMul(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        // Multiply the current value and the new value
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(val * __int_as_float(assumed)));
    } while (assumed != old);  // Repeat if another thread updated the memory
    return __int_as_float(old);
}

__global__ void sumReductionKernel(const float* __restrict__ data, float* result,
                                   const TensorLayout layout, size_t blockCount,
                                   const TensorLayout outLayout, size_t outCount) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= blockCount * outCount) return;

    size_t outIdx = physicalOffsetCUDA(i / blockCount, outLayout);
    size_t dataIdx = physicalOffsetCUDA(i, layout);

    atomicAdd(&result[outIdx], data[dataIdx]);
}

__global__ void minReductionKernel(const float* __restrict__ data, float* result,
                                   const TensorLayout layout, size_t blockCount,
                                   const TensorLayout outLayout, size_t outCount) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= blockCount * outCount) return;

    size_t outIdx = physicalOffsetCUDA(i / blockCount, outLayout);
    size_t dataIdx = physicalOffsetCUDA(i, layout);

    atomicMin(&result[outIdx], data[dataIdx]);
}

__global__ void maxReductionKernel(const float* __restrict__ data, float* result,
                                   const TensorLayout layout, size_t blockCount,
                                   const TensorLayout outLayout, size_t outCount) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= blockCount * outCount) return;

    size_t outIdx = physicalOffsetCUDA(i / blockCount, outLayout);
    size_t dataIdx = physicalOffsetCUDA(i, layout);

    atomicMax(&result[outIdx], data[dataIdx]);
}


__global__ void prodReductionKernel(const float* __restrict__ data, float* result,
                                   const TensorLayout layout, size_t blockCount,
                                   const TensorLayout outLayout, size_t outCount) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= blockCount * outCount) return;

    size_t outIdx = physicalOffsetCUDA(i / blockCount, outLayout);
    size_t dataIdx = physicalOffsetCUDA(i, layout);

    atomicMul(&result[outIdx], data[dataIdx]);
}