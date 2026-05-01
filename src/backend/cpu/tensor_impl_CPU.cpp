#include "tensor_impl_CPU.h"

#include <algorithm>
#include <random>

#include "nforge/core/tensor.h"

Tensor::CPUImpl::CPUImpl(const Tensor::Shape& shape)
    : m_shape(shape) {
    m_data.assign(m_shape.getNumElements(), 0.0f);
}

Tensor::CPUImpl::CPUImpl(const Tensor::Shape& shape, float value)
    : m_shape(shape) {
    m_data.assign(m_shape.getNumElements(), value);
}

Tensor::CPUImpl::~CPUImpl() {
    m_data.clear();
    m_data.shrink_to_fit();
}

void Tensor::CPUImpl::fillAll(float value) {
    m_data.assign(m_data.size(), value);
}

void Tensor::CPUImpl::fillRand() {
    static std::mt19937 engine(std::random_device{}());
    static std::uniform_real_distribution<double> dist(-1.0, 1.0);

    auto gen = [&]() {
        return dist(engine);
    };

    std::generate(m_data.begin(), m_data.end(), gen);
}

void Tensor::CPUImpl::print() const {
    std::cout << "====================\n";
    std::cout << "Tensor[CPU], Data:\n";

    std::vector<size_t> numElementsInDimsCurAndBelow(m_shape.getNumDims(), 1);
    for (int i = static_cast<int>(m_shape.getNumDims()) - 1; i >= 0; i--) {
        numElementsInDimsCurAndBelow[i] *= m_shape.getDim(i);
        if (i != static_cast<int>(m_shape.getNumDims()) - 1) {
            numElementsInDimsCurAndBelow[i] *= numElementsInDimsCurAndBelow[i + 1];
        }
    }

    for (size_t i = 0; i < m_data.size(); i++) {
        std::cout << m_data[i] << " ";
        for (size_t j = 0; j < m_shape.getNumDims(); j++) {
            // if element count of the block represented by suffix starting at j, divides i, then print a new line
            if (i % numElementsInDimsCurAndBelow[j] == numElementsInDimsCurAndBelow[j] - 1 && i != m_data.size() - 1) {
                std::cout << "\n";
            }
        }
    }

    std::cout << "Shape: " << getShape().toString() << "\n";
    std::cout << "====================\n";
}

void Tensor::CPUImpl::print(const std::vector<size_t>& position) const {
    std::cout << "====================\n";
    std::cout << "Tensor[CPU], Data:\n";

    std::vector<size_t> numElementsInDimsCurAndBelow(m_shape.getNumDims(), 1);
    for (int i = static_cast<int>(m_shape.getNumDims()) - 1; i >= 0; i--) {
        numElementsInDimsCurAndBelow[i] *= m_shape.getDim(i);
        if (i != static_cast<int>(m_shape.getNumDims()) - 1) {
            numElementsInDimsCurAndBelow[i] *= numElementsInDimsCurAndBelow[i + 1];
        }
    }

    size_t blockSize = m_shape.getSlice(position.size(), m_shape.getNumDims()).getNumElements();

    size_t offsetCount = blockSize;
    for (size_t i = 0; i < position.size(); i++) {
        // if first slice in a dim
        if (offsetCount == 0) {
            offsetCount = blockSize;
        }

        offsetCount *= position[i];
    }

    for (size_t i = offsetCount; i < offsetCount + blockSize; i++) {
        std::cout << m_data[i] << " ";
        for (size_t j = 0; j < m_shape.getNumDims(); j++) {
            if (i % numElementsInDimsCurAndBelow[j] == numElementsInDimsCurAndBelow[j] - 1 && i != offsetCount + blockSize - 1) {
                std::cout << "\n";
            }
        }
    }
    std::cout << "\n";

    std::cout << "Shape: " << m_shape.getSlice(position.size(), m_shape.getNumDims()).toString() << "\n";
    std::cout << "====================\n";
}

Tensor::Shape Tensor::CPUImpl::getShape() const {
    return m_shape;
}

std::string Tensor::CPUImpl::toString() const {
    std::string out;

    out += "{ ";
    for (float element : m_data) {
        out += std::to_string(element) + " ";
    }
    out += "}";

    return out;
}

size_t Tensor::CPUImpl::getNumElements() const {
    size_t numImpliedByShape = m_shape.getNumElements();
    size_t numInContainer = m_data.size();

    assert(numImpliedByShape == numInContainer);

    return numInContainer;
}

std::vector<float> Tensor::CPUImpl::toVector() const {
    return m_data;
}

float* Tensor::CPUImpl::dataPtr() const {
    return (float*)m_data.data();
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::clone() const {
    return std::make_unique<CPUImpl>(*this);
}

void Tensor::CPUImpl::set(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                          const TensorLayout& rhsLayout) {
    
    const auto* rhs = static_cast<const Tensor::CPUImpl*>(rhsImpl);

    float*       a = dataPtr();
    const float* b = rhs->dataPtr();

    size_t count = 1;
    for (size_t d = 0; d < lhsLayout.rank; d++) count *= lhsLayout.shape[d];

    for (size_t i = 0; i < count; i++) {
        a[physicalOffset(i, lhsLayout)] = b[physicalOffset(i, rhsLayout)];
    }
}

bool Tensor::CPUImpl::compare(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                              const TensorLayout& rhsLayout) const {

    const auto* rhs = static_cast<const Tensor::CPUImpl*>(rhsImpl);

    const float* a = dataPtr();
    const float* b = rhs->dataPtr();

    size_t count = 1;
    for (size_t d = 0; d < lhsLayout.rank; d++) count *= lhsLayout.shape[d];

    for (size_t i = 0; i < count; i++) {
        if (a[physicalOffset(i, lhsLayout)] != b[physicalOffset(i, rhsLayout)]) return false;
    }

    return true;
}

///////////////////////////////////////////
// Element wise binary tensor operations //
///////////////////////////////////////////


template <typename BinaryOp>
std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::applyBinaryOp(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                                                             const TensorLayout& rhsLayout, const TensorLayout& outLayout, BinaryOp op) const {

    auto outShape = Tensor::Shape(outLayout);

    auto* result = new Tensor::CPUImpl(outShape);
    const auto* rhs = static_cast<const Tensor::CPUImpl*>(rhsImpl);

    const float* a = dataPtr();
    const float* b = rhs->dataPtr();
    auto&        c = result->m_data;

    size_t count = 1;
    for (size_t d = 0; d < outLayout.rank; d++) count *= outLayout.shape[d];

    for (size_t i = 0; i < count; i++) {
        c[physicalOffset(i, outLayout)] =
            op(a[physicalOffset(i, lhsLayout)],
               b[physicalOffset(i, rhsLayout)]);
    }
    return std::unique_ptr<Tensor::Impl>(result);
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::add(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
                                                   const TensorLayout& rhsLayout, const TensorLayout& outLayout) const {
    return applyBinaryOp(lhsLayout, rhsImpl, rhsLayout, outLayout, [](float a, float b) {
        return a + b;
    });
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::sub(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
                                                   const TensorLayout& rhsLayout, const TensorLayout& outLayout) const {
    return applyBinaryOp(lhsLayout, rhsImpl, rhsLayout, outLayout, [](float a, float b) {
        return a - b;
    });
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::mul(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
                                                   const TensorLayout& rhsLayout, const TensorLayout& outLayout) const {
    return applyBinaryOp(lhsLayout, rhsImpl, rhsLayout, outLayout, [](float a, float b) {
        return a * b;
    });
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::div(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
                                                   const TensorLayout& rhsLayout, const TensorLayout& outLayout) const {
    return applyBinaryOp(lhsLayout, rhsImpl, rhsLayout, outLayout, [](float a, float b) {
        return a / b;
    });
}

template <typename BinaryOp>
void Tensor::CPUImpl::applyInplaceBinaryOp(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                                           const TensorLayout& rhsLayout, BinaryOp op) {

    const auto* rhs = static_cast<const Tensor::CPUImpl*>(rhsImpl);

    float*       a = dataPtr();
    const float* b = rhs->dataPtr();

    size_t count = 1;
    for (size_t d = 0; d < lhsLayout.rank; d++) count *= lhsLayout.shape[d];

    for (size_t i = 0; i < count; i++) {
        a[physicalOffset(i, lhsLayout)] =
            op(a[physicalOffset(i, lhsLayout)],
               b[physicalOffset(i, rhsLayout)]);
    }
}

void Tensor::CPUImpl::iadd(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, const TensorLayout& rhsLayout) {
    applyInplaceBinaryOp(lhsLayout, rhsImpl, rhsLayout, [](float a, float b) {
        return a + b;
    });
}

void Tensor::CPUImpl::isub(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, const TensorLayout& rhsLayout) {
    applyInplaceBinaryOp(lhsLayout, rhsImpl, rhsLayout, [](float a, float b) {
        return a - b;
    });
}

void Tensor::CPUImpl::imul(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, const TensorLayout& rhsLayout) {
    applyInplaceBinaryOp(lhsLayout, rhsImpl, rhsLayout, [](float a, float b) {
        return a * b;
    });
}

void Tensor::CPUImpl::idiv(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, const TensorLayout& rhsLayout) {
    applyInplaceBinaryOp(lhsLayout, rhsImpl, rhsLayout, [](float a, float b) {
        return a / b;
    });
}

template <typename ReductionOp>
std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::applyReductionOp(const TensorLayout& layout, const TensorLayout& blockLayout,
                                                                const TensorLayout& outLayout, ReductionOp op) const {
    auto outShape = Tensor::Shape(outLayout);

    auto* result = new Tensor::CPUImpl(outShape);

    const float* a = dataPtr();
    auto&        b = result->m_data;

    size_t outCount = 1;
    for (size_t d = 0; d < outLayout.rank; d++) outCount *= outLayout.shape[d];

    size_t blockCount = 1;
    for (size_t d = 0; d < blockLayout.rank; d++) blockCount *= blockLayout.shape[d];

    for (size_t i = 0; i < outCount; i++) {
        float res = a[physicalOffset(i * blockCount, layout)];

        for (size_t j = 1; j < blockCount; j++) {
            res = op(res, a[physicalOffset(i * blockCount + j, layout)]);
        }

        b[physicalOffset(i, outLayout)] = res;
    }
    return std::unique_ptr<Tensor::Impl>(result);
}


std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::sum(const TensorLayout& layout, const TensorLayout& blockLayout, 
                                                    const TensorLayout& outLayout) const {
    return applyReductionOp(layout, blockLayout, outLayout, [](float a, float b) {
        return a + b;
    });
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::min(const TensorLayout& layout, const TensorLayout& blockLayout, 
                                                   const TensorLayout& outLayout) const {
    return applyReductionOp(layout, blockLayout, outLayout, [](float a, float b) {
        return std::min(a, b);
    });
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::max(const TensorLayout& layout, const TensorLayout& blockLayout, 
                                                   const TensorLayout& outLayout) const {
    return applyReductionOp(layout, blockLayout, outLayout, [](float a, float b) {
        return std::max(a, b);
    });
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::prod(const TensorLayout& layout, const TensorLayout& blockLayout, 
                                                    const TensorLayout& outLayout) const {
    return applyReductionOp(layout, blockLayout, outLayout, [](float a, float b) {
        return a * b;
    });
}
    