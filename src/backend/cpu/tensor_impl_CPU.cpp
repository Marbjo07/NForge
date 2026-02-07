#include "nforge/backend/cpu/tensor_impl_CPU.h"

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

    std::cout << "Shape: " << shape().toString() << "\n";
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

Tensor::Shape Tensor::CPUImpl::shape() const {
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

size_t Tensor::CPUImpl::numElements() const {
    size_t numImpliedByShape = m_shape.getNumElements();
    size_t numInContainer = m_data.size();

    assert(numImpliedByShape == numInContainer);

    return numInContainer;
}

std::vector<float> Tensor::CPUImpl::toVector() const {
    return m_data;
}

const float* Tensor::CPUImpl::data() const {
    return m_data.data();
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::clone() const {
    return std::make_unique<CPUImpl>(*this);
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::get(size_t idx) const {
    // shape {3, 2, 5}
    //[[[x, x, x, x, x], [x, x, x, x, x]],
    // [[x, x, x, x, x], [x, x, x, x, x]],
    // [[x, x, x, x, x], [x, x, x, x, x]]]

    if (m_shape.getNumDims() == 0) {
        throw std::runtime_error("Can not index tensor of zero dimensions");
        return std::unique_ptr<Tensor::Impl>(new Tensor::CPUImpl(*this));
    }

    if (m_shape.getNumDims() == 1 && m_shape.getDim(0) == 1) {
        throw std::runtime_error("Can not index tensor of one dimension and one element");
        return std::unique_ptr<Tensor::Impl>(new Tensor::CPUImpl(*this));
    }

    Tensor::Shape newShape = m_shape.getSlice(1, m_shape.getNumDims());
    Tensor::CPUImpl* results = new Tensor::CPUImpl(newShape);

    size_t offset = newShape.getNumElements() * idx;
    for (size_t i = 0; i < newShape.getNumElements(); i++) {
        results->m_data[i] = m_data[i + offset];
    }

    return std::unique_ptr<Tensor::Impl>(results);
}

void Tensor::CPUImpl::set(size_t lhsOffset, const Tensor::Impl& rhs, size_t rhsOffset, size_t count) {
    const auto& o = static_cast<const Tensor::CPUImpl&>(rhs);

    float* a = m_data.data() + lhsOffset;
    const float* b = rhs.data() + rhsOffset;

    for (size_t i = 0; i < count; i++) {
        a[i] = b[i];
    }
}

bool Tensor::CPUImpl::compare(size_t lhsOffset, const Tensor::Impl& rhs, size_t rhsOffset, size_t count) const {
    const auto& o = static_cast<const Tensor::CPUImpl&>(rhs);

    const float* a = m_data.data() + lhsOffset;
    const float* b = rhs.data() + rhsOffset;

    for (size_t i = 0; i < count; i++) {
        if (a[i] != b[i]) return false;
    }

    return true;
}

///////////////////////////////////////////
// Element wise binary tensor operations //
///////////////////////////////////////////

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::add(const Tensor::Impl& other) const {
    Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape.toVector());

    const Tensor::CPUImpl* o = dynamic_cast<const Tensor::CPUImpl*>(&other);

    for (size_t i = 0; i < m_data.size(); i++) {
        results->m_data[i] = m_data[i] + o->m_data[i];
    }

    return std::unique_ptr<Tensor::Impl>(results);
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::sub(const Tensor::Impl& other) const {
    Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape.toVector());

    const Tensor::CPUImpl* o = dynamic_cast<const Tensor::CPUImpl*>(&other);

    for (size_t i = 0; i < m_data.size(); i++) {
        results->m_data[i] = m_data[i] - o->m_data[i];
    }

    return std::unique_ptr<Tensor::Impl>(results);
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::mul(const Tensor::Impl& other) const {
    Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape.toVector());

    const Tensor::CPUImpl* o = dynamic_cast<const Tensor::CPUImpl*>(&other);

    for (size_t i = 0; i < m_data.size(); i++) {
        results->m_data[i] = m_data[i] * o->m_data[i];
    }

    return std::unique_ptr<Tensor::Impl>(results);
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::div(const Tensor::Impl& other) const {
    Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape.toVector());

    const Tensor::CPUImpl* o = dynamic_cast<const Tensor::CPUImpl*>(&other);

    for (size_t i = 0; i < m_data.size(); i++) {
        results->m_data[i] = m_data[i] / o->m_data[i];
    }

    return std::unique_ptr<Tensor::Impl>(results);
}

//////////////////////////////////////////////////
// Element wise binary tensor-scalar operations //
//////////////////////////////////////////////////

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::addScalar(const Tensor::Impl& other) const {
    Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape.toVector());

    float scalar = other.toVector()[0];

    for (size_t i = 0; i < m_data.size(); i++) {
        results->m_data[i] = m_data[i] + scalar;
    }

    return std::unique_ptr<Tensor::Impl>(results);
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::subScalar(const Tensor::Impl& other) const {
    Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape.toVector());

    float scalar = other.toVector()[0];

    for (size_t i = 0; i < m_data.size(); i++) {
        results->m_data[i] = m_data[i] - scalar;
    }

    return std::unique_ptr<Tensor::Impl>(results);
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::mulScalar(const Tensor::Impl& other) const {
    Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape.toVector());

    float scalar = other.toVector()[0];

    for (size_t i = 0; i < m_data.size(); i++) {
        results->m_data[i] = m_data[i] * scalar;
    }

    return std::unique_ptr<Tensor::Impl>(results);
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::divScalar(const Tensor::Impl& other) const {
    Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape.toVector());

    float scalar = other.toVector()[0];

    for (size_t i = 0; i < m_data.size(); i++) {
        results->m_data[i] = m_data[i] / scalar;
    }

    return std::unique_ptr<Tensor::Impl>(results);
}
