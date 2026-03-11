#include "nforge/backend/cuda/tensor_impl_CUDA.h"

#include <algorithm>
#include <random>
#include <sstream>
#include <iostream>

#include "nforge/core/tensor.h"
#include "backend/cuda/kernels/kernels.cuh"
#include "backend/cuda/utils/cuda_error.h"
#include "backend/cuda/utils/cuda_context.h"

#include <cuda_runtime.h>


constexpr size_t SCALAR_READ_OFFSET = 0;


Tensor::CUDAImpl::CUDAImpl(const Tensor::Shape& shape) 
    : m_shape(shape) {
    size_t numElements = shape.getNumElements();
    CUDA_CHECK(cudaMalloc((void**)&d_data, numElements * sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
}

Tensor::CUDAImpl::~CUDAImpl() {
    cudaFree(d_data);
}

void Tensor::CUDAImpl::fillAll(float value) {
    int threads = 256;
    int blocks  = max(1, ((int)m_shape.getNumElements() + threads - 1) / threads);
    fillKernel<<<blocks, threads>>>(d_data, value, (unsigned int)m_shape.getNumElements());
}

void Tensor::CUDAImpl::fillRand() {
    std::cout << "not implemented\n";
}

void Tensor::CUDAImpl::print() const {
    std::cout << toString() << "\n";
}

void Tensor::CUDAImpl::print(const std::vector<size_t>& position) const {
    std::cout << "Not implemented\n";
}

size_t Tensor::CUDAImpl::getNumElements() const {
    return m_shape.getNumElements();
}

Tensor::Shape Tensor::CUDAImpl::getShape() const {
    return m_shape;
}

float* Tensor::CUDAImpl::dataPtr() const {
    return d_data;
}

std::vector<float> Tensor::CUDAImpl::toVector() const {
    std::vector<float> result(m_shape.getNumElements());

    // sync all operations
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(CudaContext::get().stream()));
    CUDA_CHECK(cudaMemcpy(result.data(), d_data,
                          m_shape.getNumElements() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaGetLastError());
    return result;
}

std::string Tensor::CUDAImpl::toString() const {
    std::vector<float> data = toVector();

    std::string out;

    out += "{ ";
    for (float element : data) {
        out += std::to_string(element) + " ";
    }
    out += "}";

    return out;
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::clone() const {
    CUDAImpl* copy = new CUDAImpl(m_shape);
    
    // sync
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(CudaContext::get().stream()));

    CUDA_CHECK(cudaMemcpy(copy->d_data, d_data,
                          m_shape.getNumElements() * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaGetLastError());
    return std::unique_ptr<Tensor::Impl>(copy);
}

// Assignments and indexing
void Tensor::CUDAImpl::set(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) {
    std::cout << "not implemented\n";
}

// Comparisons
bool Tensor::CUDAImpl::compare(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const {
    const Tensor::CUDAImpl* o = cast(rhs);

    // init equal flag
    int h_equalFlag = 1; 
    int* d_equalFlag;
    cudaMalloc(&d_equalFlag, sizeof(int));
    cudaMemcpy(d_equalFlag, &h_equalFlag, 1, cudaMemcpyHostToDevice);

    // get all data pointers
    const float* lhsDataPtr = dataPtr() + lhsOffset;
    const float* rhsDataPtr = o->dataPtr() + rhsOffset;

    // launch kernel
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    checkAllEqualKernel<<<blocks, threads>>>(lhsDataPtr, rhsDataPtr, d_equalFlag, count);
    CUDA_CHECK(cudaGetLastError());


    // copy flag from device
    cudaMemcpy(&h_equalFlag, d_equalFlag, 1, cudaMemcpyDeviceToHost);

    return h_equalFlag;
}



template<typename Kernel>
std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::applyKernel(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count, Kernel kernel) const {
    // create output tensor
    Tensor::CUDAImpl* results = new Tensor::CUDAImpl(this->m_shape.toVector());

    const Tensor::CUDAImpl* o = cast(rhs);

    // get all data pointers
    const float* lhsDataPtr = dataPtr() + lhsOffset;
    const float* rhsDataPtr = o->dataPtr() + rhsOffset;
    float* resultsDataPtr = results->dataPtr();

    // launch kernel
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    kernel<<<blocks, threads>>>(lhsDataPtr, rhsDataPtr, resultsDataPtr, count);
    CUDA_CHECK(cudaGetLastError());

    return std::unique_ptr<Tensor::Impl>(results);
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::add(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const {
    return applyKernel(lhsOffset, rhs, rhsOffset, count, addKernel);
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::sub(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const {
    return applyKernel(lhsOffset, rhs, rhsOffset, count, subKernel);
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::mul(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const {
    return applyKernel(lhsOffset, rhs, rhsOffset, count, mulKernel);
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::div(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const {
    return applyKernel(lhsOffset, rhs, rhsOffset, count, divKernel);
}


std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::addScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const {
    return applyKernel(lhsOffset, rhs, SCALAR_READ_OFFSET, count, addKernel);
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::subScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const {
    return applyKernel(lhsOffset, rhs, SCALAR_READ_OFFSET, count, subKernel);
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::mulScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const {
    return applyKernel(lhsOffset, rhs, SCALAR_READ_OFFSET, count, mulKernel);
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::divScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const {
    return applyKernel(lhsOffset, rhs, SCALAR_READ_OFFSET, count, divKernel);
}

const Tensor::CUDAImpl* Tensor::CUDAImpl::cast(const Tensor::Impl* p) const {
    return static_cast<const Tensor::CUDAImpl*>(p);
}