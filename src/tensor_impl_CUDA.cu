#include "pch.h"

#include "nforge/tensor.h"
#include "nforge/tensor_backend.h"
#include "nforge/error_handler.h"

#ifdef BUILD_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif // BUILD_WITH_CUDA

__global__
void addition_kernel(float* a, float* b, float* c, size_t size) {
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x; 
	
	for (size_t i = index; i < size; i += stride) { 
		c[i] = a[i] + b[i]; 
	}
}

__global__
void subtraction_kernel(float* a, float* b, float* c, size_t size) {
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;

	for (size_t i = index; i < size; i += stride) {
		c[i] = a[i] - b[i];
	}
}

__global__
void multiplication_kernel(float* a, float* b, float* c, size_t size) {
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;

	for (size_t i = index; i < size; i += stride) {
		c[i] = a[i] * b[i];
	}
}

__global__
void division_kernel(float* a, float* b, float* c, size_t size) {
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;

	for (size_t i = index; i < size; i += stride) {
		c[i] = a[i] / b[i];
	}
}

Tensor backend::cuda::pointwise::addition(const Tensor& a, const Tensor& b) {
	if (a.m_size != b.m_size) {
		LOG(ERROR) << "Mismatch between tensors when adding!";
		return a;
	}

	Tensor result(a.m_shape, Backend::CUDA);
		
	size_t block_size = 256;
	size_t num_blocks = (a.m_size + block_size - 1) / block_size;
	addition_kernel << <num_blocks, block_size >> > (a.m_data, b.m_data, result.m_data, a.m_size);

	return result;
}

Tensor backend::cuda::pointwise::subtraction(const Tensor& a, const Tensor& b) {
	if (a.m_size != b.m_size) {
		LOG(ERROR) << "Mismatch between tensors when subtracting!";
		return a;
	}

	Tensor result(a.m_shape, Backend::CUDA);

	size_t block_size = 256;
	size_t num_blocks = (a.m_size + block_size - 1) / block_size;
	subtraction_kernel << <num_blocks, block_size >> > (a.m_data, b.m_data, result.m_data, a.m_size);

	return result;
}

Tensor backend::cuda::pointwise::multiplication(const Tensor& a, const Tensor& b) {
	if (a.m_size != b.m_size) {
		LOG(ERROR) << "Mismatch between tensors when multiplying!";
		return a;
	}

	Tensor result(a.m_shape, Backend::CUDA);

	size_t block_size = 256;
	size_t num_blocks = (a.m_size + block_size - 1) / block_size;
	multiplication_kernel << <num_blocks, block_size >> > (a.m_data, b.m_data, result.m_data, a.m_size);
	
	return result;
}

Tensor backend::cuda::pointwise::division(const Tensor& a, const Tensor& b) {
	if (a.m_size != b.m_size) {
		LOG(ERROR) << "Mismatch between tensors when dividing!";
		return a;
	}

	Tensor result(a.m_shape, Backend::CUDA);

	size_t block_size = 256;
	size_t num_blocks = (a.m_size + block_size - 1) / block_size;
	division_kernel << <num_blocks, block_size >> > (a.m_data, b.m_data, result.m_data, a.m_size);

	return result;
}
