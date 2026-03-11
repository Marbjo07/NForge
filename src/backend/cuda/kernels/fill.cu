#include "backend/cuda/kernels/kernels.cuh"

__global__ void fillKernel(float* __restrict__ data, float value, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) data[i] = value;
}