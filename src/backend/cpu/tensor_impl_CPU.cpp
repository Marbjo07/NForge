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

void Tensor::CPUImpl::set(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) {
    const Tensor::CPUImpl* o = static_cast<const Tensor::CPUImpl*>(rhs);

    float* a = dataPtr() + lhsOffset;
    const float* b = o->dataPtr() + rhsOffset;

    for (size_t i = 0; i < count; i++) {
        a[i] = b[i];
    }
}

bool Tensor::CPUImpl::compare(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const {
    const Tensor::CPUImpl* o = static_cast<const Tensor::CPUImpl*>(rhs);

    const float* a = dataPtr() + lhsOffset;
    const float* b = o->dataPtr() + rhsOffset;

    for (size_t i = 0; i < count; i++) {
        if (a[i] != b[i]) return false;
    }

    return true;
}

///////////////////////////////////////////
// Element wise binary tensor operations //
///////////////////////////////////////////

static inline size_t physicalOffset(size_t linear, const TensorLayout& L) {
    size_t off = L.offset;
    for (int d = L.rank - 1; d >= 0; d--) {
        size_t coord = linear % L.shape[d];
        linear /= L.shape[d];
        off += coord * L.strides[d];
    }
    return off;
}

template <typename BinaryOp>
std::unique_ptr<Tensor::Impl> applyBinaryOp(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
    const TensorLayout& rhsLayout, const TensorLayout& outLayout, BinaryOp op) const {

    auto* result = new Tensor::CPUImpl(out);
    const auto* rhs = static_cast<const Tensor::CPUImpl*>(rhsImpl);

    const float* a = dataPtr() + lhsOffset;
    const float* b = o->dataPtr() + rhsOffset;
    float*       c = result->m_data;

    size_t count = 1;
    for (int d = 0; d < out.rank; d++) count *= out.shape[d];

    for (size_t i = 0; i < count; i++) {
        c[physicalOffset(i, out)] =
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