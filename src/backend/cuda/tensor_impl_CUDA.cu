#include "tensor_impl_CUDA.h"

#include "backend/cuda/kernels/kernels.cuh"
#include "backend/cuda/utils/cuda_context.h"

#include <cuda_runtime.h>


Tensor::CUDAImpl::CUDAImpl(const Tensor::Shape& shape)
    : m_shape(shape) {
    size_t numElements = shape.getNumElements();
    CUDA_CHECK(cudaMalloc((void**)&d_data, numElements * sizeof(float)));
    CUDA_CHECK(cudaMemset((void**)d_data, 0, numElements * sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
}

Tensor::CUDAImpl::~CUDAImpl() {
    cudaFree(d_data);
}

void Tensor::CUDAImpl::fillAll(float value) {
    int threads = 256;
    int blocks = std::max(1, ((int)m_shape.getNumElements() + threads - 1) / threads);
    fillKernel<<<blocks, threads, 0, CudaContext::get().stream()>>>(d_data, value, (unsigned int)m_shape.getNumElements());
}

void Tensor::CUDAImpl::fillRand() {
    size_t numElements = m_shape.getNumElements();
    CURAND_CHECK(curandGenerateUniform(CudaContext::get().rng(), d_data, numElements));
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
void Tensor::CUDAImpl::set(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
                           const TensorLayout& rhsLayout) {
    const Tensor::CUDAImpl* o = cast(rhsImpl);

    float* a = dataPtr();
    float* b = o->dataPtr();

    size_t count = 1;
    for (size_t d = 0; d < rhsLayout.rank; d++) count *= rhsLayout.shape[d];

    int threads = 256;
    int blocks = ((int)count + threads - 1) / threads;
    setKernel<<<blocks, threads, 0, CudaContext::get().stream()>>>(a, lhsLayout, b, rhsLayout, (int)count);
    CUDA_CHECK(cudaGetLastError());
}

// Comparisons
bool Tensor::CUDAImpl::compare(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
                               const TensorLayout& rhsLayout) const {
    const Tensor::CUDAImpl* o = cast(rhsImpl);

    // init equal flag
    int h_equalFlag = 1;
    int* d_equalFlag;
    cudaMalloc(&d_equalFlag, sizeof(int));
    cudaMemcpy(d_equalFlag, &h_equalFlag, sizeof(int), cudaMemcpyHostToDevice);

    // get all data pointers
    const float* lhsDataPtr = dataPtr();
    const float* rhsDataPtr = o->dataPtr();

    size_t count = 1;
    for (size_t d = 0; d < rhsLayout.rank; d++) count *= rhsLayout.shape[d];

    // launch kernel
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    checkAllEqualKernel<<<blocks, threads, 0, CudaContext::get().stream()>>>(lhsDataPtr, lhsLayout, rhsDataPtr, rhsLayout, d_equalFlag, count);
    CUDA_CHECK(cudaGetLastError());

    // copy flag from device
    cudaMemcpy(&h_equalFlag, d_equalFlag, sizeof(int), cudaMemcpyDeviceToHost);

    return h_equalFlag;
}

const Tensor::CUDAImpl* Tensor::CUDAImpl::cast(const Tensor::Impl* p) const {
    return static_cast<const Tensor::CUDAImpl*>(p);
}

template <typename Kernel>
std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::applyKernel(
    const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
    const TensorLayout& rhsLayout, const TensorLayout& outLayout, Kernel kernel) const {
    // create output tensor
    auto outShape = Tensor::Shape(outLayout);
    auto* results = new Tensor::CUDAImpl(outLayout);

    const Tensor::CUDAImpl* o = cast(rhsImpl);

    // get all data pointers
    const float* lhs = dataPtr();
    const float* rhs = o->dataPtr();
    float* out = results->dataPtr();

    size_t count = 1;
    for (size_t d = 0; d < outLayout.rank; d++) count *= outLayout.shape[d];

    // launch kernel
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    kernel<<<blocks, threads, 0, CudaContext::get().stream()>>>(lhs, lhsLayout, rhs, rhsLayout, out, outLayout, count);
    CUDA_CHECK(cudaGetLastError());

    return std::unique_ptr<Tensor::Impl>(results);
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::add(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
                                                    const TensorLayout& rhsLayout, const TensorLayout& outLayout) const {
    return applyKernel(lhsLayout, rhsImpl, rhsLayout, outLayout, addKernel);
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::sub(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
                                                    const TensorLayout& rhsLayout, const TensorLayout& outLayout) const {
    return applyKernel(lhsLayout, rhsImpl, rhsLayout, outLayout, subKernel);
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::mul(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
                                                    const TensorLayout& rhsLayout, const TensorLayout& outLayout) const {
    return applyKernel(lhsLayout, rhsImpl, rhsLayout, outLayout, mulKernel);
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::div(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
                                                    const TensorLayout& rhsLayout, const TensorLayout& outLayout) const {
    return applyKernel(lhsLayout, rhsImpl, rhsLayout, outLayout, divKernel);
}

template <typename Kernel>
void Tensor::CUDAImpl::applyInplaceKernel(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
                                          const TensorLayout& rhsLayout, Kernel kernel) {
    const Tensor::CUDAImpl* o = cast(rhsImpl);

    // get all data pointers
    float* lhs = dataPtr();
    const float* rhs = o->dataPtr();

    size_t count = 1;
    for (size_t d = 0; d < lhsLayout.rank; d++) count *= lhsLayout.shape[d];

    // launch kernel
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    kernel<<<blocks, threads, 0, CudaContext::get().stream()>>>(lhs, lhsLayout, rhs, rhsLayout, count);
    CUDA_CHECK(cudaGetLastError());
}

void Tensor::CUDAImpl::iadd(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, const TensorLayout& rhsLayout) {
    applyInplaceKernel(lhsLayout, rhsImpl, rhsLayout, iaddKernel);
}

void Tensor::CUDAImpl::isub(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, const TensorLayout& rhsLayout) {
    applyInplaceKernel(lhsLayout, rhsImpl, rhsLayout, isubKernel);
}

void Tensor::CUDAImpl::imul(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, const TensorLayout& rhsLayout) {
    applyInplaceKernel(lhsLayout, rhsImpl, rhsLayout, imulKernel);
}

void Tensor::CUDAImpl::idiv(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, const TensorLayout& rhsLayout) {
    applyInplaceKernel(lhsLayout, rhsImpl, rhsLayout, idivKernel);
}

template <typename Kernel>
std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::applyReductionKernel(const TensorLayout& layout, const TensorLayout& blockLayout,
                                                                     const TensorLayout& outLayout, float initValue, Kernel kernel) const {
    // create output tensor
    auto outShape = Tensor::Shape(outLayout);
    auto* results = new Tensor::CUDAImpl(outShape);

    const float* lhs = dataPtr();
    float* out = results->dataPtr();

    // number of elements in output tensor
    size_t outCount = 1;
    for (size_t d = 0; d < outLayout.rank; d++) outCount *= outLayout.shape[d];

    // size of each block to be reduced
    size_t blockCount = 1;
    for (size_t d = 0; d < blockLayout.rank; d++) blockCount *= blockLayout.shape[d];

    size_t count = blockCount * outCount;

    // initialize output buffer to initValue, then call reduction kernel
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    fillKernel<<<blocks, threads, 0, CudaContext::get().stream()>>>(out, initValue, outCount);
    kernel<<<blocks, threads, 0, CudaContext::get().stream()>>>(lhs, out, layout, blockCount, outLayout, outCount);
    CUDA_CHECK(cudaGetLastError());

    return std::unique_ptr<Tensor::Impl>(results);
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::sum(const TensorLayout& layout, const TensorLayout& blockLayout,
                                                    const TensorLayout& outLayout) const {
    return applyReductionKernel(layout, blockLayout, outLayout, 0.0f, sumReductionKernel);
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::min(const TensorLayout& layout, const TensorLayout& blockLayout,
                                                    const TensorLayout& outLayout) const {
    return applyReductionKernel(layout, blockLayout, outLayout, FLT_MAX, minReductionKernel);
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::max(const TensorLayout& layout, const TensorLayout& blockLayout,
                                                    const TensorLayout& outLayout) const {
    return applyReductionKernel(layout, blockLayout, outLayout, -FLT_MAX, maxReductionKernel);
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::prod(const TensorLayout& layout, const TensorLayout& blockLayout,
                                                     const TensorLayout& outLayout) const {
    return applyReductionKernel(layout, blockLayout, outLayout, 1.0f, prodReductionKernel);
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::norm(const TensorLayout& layout) const {
    // create output tensor
    auto outShape = Tensor::Shape({1});
    auto* results = new Tensor::CUDAImpl(outShape);

    // get all data pointers
    const float* lhs = dataPtr();
    float* out = results->dataPtr();

    size_t count = 1;
    for (size_t d = 0; d < layout.rank; d++) count *= layout.shape[d];

    // get sum of all elements squared
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    squareSumKernel<<<blocks, threads, 0, CudaContext::get().stream()>>>(lhs, out, count);
    CUDA_CHECK(cudaGetLastError());

    // apply sqrt to out
    isqrtKernel<<<1, 1, 0, CudaContext::get().stream()>>>(out, 1);
    CUDA_CHECK(cudaGetLastError());

    return std::unique_ptr<Tensor::Impl>(results);
}
