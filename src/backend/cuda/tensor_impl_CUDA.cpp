#include "nforge/backend/cuda/tensor_impl_CUDA.h"

#include <algorithm>
#include <random>

#include "nforge/core/tensor.h"

Tensor::CUDAImpl::CUDAImpl(const Tensor::Shape& shape)
    : m_shape(shape) {
    //m_data.assign(m_shape.getNumElements(), 0.0f);
}

Tensor::CUDAImpl::CUDAImpl(const Tensor::Shape& shape, float value)
    : m_shape(shape) {
    //m_data.assign(m_shape.getNumElements(), value);
}

Tensor::CUDAImpl::~CUDAImpl() {
    //m_data.clear();
    //m_data.shrink_to_fit();
}

void Tensor::CUDAImpl::fillAll(float value) {
    throw std::runtime_error("Can't fill cuda tensor with values");
}

void Tensor::CUDAImpl::fillRand() {
    throw std::runtime_error("Can't fill cuda tensor with random values");
}

void Tensor::CUDAImpl::print() const {
    throw std::runtime_error("Can't print cuda tensor");
}

void Tensor::CUDAImpl::print(const std::vector<size_t>& position) const {
    throw std::runtime_error("Can't print cuda tensor");
}

Tensor::Shape Tensor::CUDAImpl::getShape() const {
    return m_shape;
}

std::string Tensor::CUDAImpl::toString() const {
    throw std::runtime_error("Can't convert cuda tensor to string");
}

size_t Tensor::CUDAImpl::getNumElements() const {
    return m_shape.getNumElements();
}

std::vector<float> Tensor::CUDAImpl::toVector() const {
    throw std::runtime_error("Can't convert cuda tensor to vector");
}

float* Tensor::CUDAImpl::dataPtr() const {
    throw std::runtime_error("Can't give data ptr for cuda tensor");
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::clone() const {
    return std::make_unique<CUDAImpl>(*this);
}

void Tensor::CUDAImpl::set(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) {
    throw std::runtime_error("Can't set values for cuda tensor");
}

bool Tensor::CUDAImpl::compare(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const {
    throw std::runtime_error("Can't compare values for cuda tensor");
}

///////////////////////////////////////////
// Element wise binary tensor operations //
///////////////////////////////////////////

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::add(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const {
    throw std::runtime_error("Can't add values for cuda tensor");
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::sub(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const {
    throw std::runtime_error("Can't subtract values for cuda tensor");
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::mul(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const {
    throw std::runtime_error("Can't multiply values for cuda tensor");
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::div(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const {
    throw std::runtime_error("Can't divide values for cuda tensor");
}

//////////////////////////////////////////////////
// Element wise binary tensor-scalar operations //
//////////////////////////////////////////////////

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::addScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const {
    throw std::runtime_error("Can't increment values by scalar for cuda tensor");
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::subScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const {
    throw std::runtime_error("Can't subtract values by scalar for cuda tensor");
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::mulScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const {
    throw std::runtime_error("Can't multiply values by scalar for cuda tensor");
}

std::unique_ptr<Tensor::Impl> Tensor::CUDAImpl::divScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const {
    throw std::runtime_error("Can't divide values by scalar for cuda tensor");
}