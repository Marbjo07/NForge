#include "backend/cuda/kernels/kernels.cuh"

__global__ void addKernel(const float* __restrict__ lhs, const float* __restrict__ rhs, float* __restrict__ out, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = lhs[i] + rhs[i]; 
}

__global__ void subKernel(const float* __restrict__ lhs, const float* __restrict__ rhs, float* __restrict__ out, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = lhs[i] - rhs[i];
}

__global__ void mulKernel(const float* __restrict__ lhs, const float* __restrict__ rhs, float* __restrict__ out, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = lhs[i] * rhs[i];
}

__global__ void divKernel(const float* __restrict__ lhs, const float* __restrict__ rhs, float* __restrict__ out, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = lhs[i] / rhs[i];
}


// Scalar ops
__global__ void addScalarKernel(const float* __restrict__ lhs, const float* __restrict__ scalar, float* __restrict__ out, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = lhs[i] + scalar[0];
}

__global__ void subScalarKernel(const float* __restrict__ lhs, const float* __restrict__ scalar, float* __restrict__ out, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = lhs[i] - scalar[0];
}

__global__ void mulScalarKernel(const float* __restrict__ lhs, const float* __restrict__ scalar, float* __restrict__ out, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = lhs[i] * scalar[0];
}

__global__ void divScalarKernel(const float* __restrict__ lhs, const float* __restrict__ scalar, float* __restrict__ out, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = lhs[i] / scalar[0];
}


__global__ void fillKernel(float* __restrict__ data, float value, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) data[i] = value;
}


__global__ void checkAllEqualKernel(const float* __restrict__ lhs, const float* __restrict__ rhs, int* isEqualFlag, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && lhs[i] != rhs[i]) *isEqualFlag = 0;
}