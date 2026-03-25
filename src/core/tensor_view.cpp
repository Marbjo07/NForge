#include "nforge/core/tensor_view.h"

Tensor::View::View(Tensor& parent, const std::vector<size_t>& index)
    : m_parent(parent), m_position(index) {
    m_shape = m_parent.getShape()[m_position];
    m_stride.assign(m_shape.getNumDims(), 1);
}

Tensor::View::View(Tensor& parent, const std::vector<size_t>& index,
                   const std::vector<size_t>& stride, const Tensor::Shape& shape)
    : m_parent(parent), m_position(index), m_stride(stride) {

    Tensor::Shape blockShape = m_parent.getShape()[m_position];

    if (m_stride.size() != shape.getNumDims()) {
        throw std::runtime_error("Stride must have same number of dimensions as view shape");
    }

    // prefix-only zeros
    bool seenNonZero = false;
    for (size_t s : m_stride) {
        if (seenNonZero && s == 0) {
            throw std::runtime_error("Zero stride only allowed as a prefix");
        }
        if (s != 0) seenNonZero = true;
    }

    // count zero-prefix length
    size_t prefixLen = 0;
    for (size_t s : m_stride) {
        if (s == 0) prefixLen++;
        else break;
    }

    // the non-zero suffix must correspond to the block shape
    size_t suffixLen = m_stride.size() - prefixLen;
    if (suffixLen != blockShape.getNumDims()) {
        throw std::runtime_error(
            "Non-zero stride dimensions (" + std::to_string(suffixLen) +
            ") must match block dimensions (" + std::to_string(blockShape.getNumDims()) + ")");
    }

    // validate suffix dims against block shape
    for (size_t i = 0; i < suffixLen; i++) {
        size_t blockDim = blockShape.getDim(i);
        size_t s = m_stride[prefixLen + i];
        size_t expected = blockDim / s;

        if (blockDim % s != 0) {
            throw std::runtime_error("Stride " + std::to_string(s) +
                " does not evenly divide block dimension " + std::to_string(blockDim));
        }
        if (shape.getDim(prefixLen + i) != expected) {
            throw std::runtime_error("Stride and block shape does not result in target shape");
        }
    }

    // zero-prefix dims: accept whatever the caller specified (no block dim to check against)

    m_shape = shape;
}

Tensor::View::View(Tensor& parent, const std::vector<size_t>& stride, const Tensor::Shape& shape, BroadcastTag)
    : m_parent(parent), m_position({}), m_stride(stride), m_shape(shape) {
}

Tensor::View::View(Tensor& parent, const std::vector<size_t>& index, const std::vector<size_t>& stride)
    : m_parent(parent), m_position(index), m_stride(stride) {
        
    Tensor::Shape blockShape = m_parent.getShape()[m_position];

    if (m_stride.size() != blockShape.getNumDims()) {
        throw std::runtime_error("Stride must have same number of dimensions as indexed block, stride: " + 
         std::to_string(m_stride.size()) + " block: " + std::to_string(blockShape.getNumDims()));
    }

    // can't contain any zeros
    for (size_t s : m_stride) {
        if (s == 0) {
            throw std::runtime_error("Zero stride only allowed without specifying shape");
        }
    }

    std::vector<size_t> dims(blockShape.getNumDims());

    // validate that the target shape is possible
    for (size_t i = 0; i < m_stride.size(); i++) {
        size_t blockDim = blockShape.getDim(i);
        if (blockDim % m_stride[i] != 0) {
            throw std::runtime_error("Stride does not match block shape, dim index: " + 
                std::to_string(i) + " block dim: " + std::to_string(blockDim) + " stride: " + std::to_string(m_stride[i]));
        }

        dims[i] = blockDim / m_stride[i];
    }


    m_shape = Tensor::Shape(dims);
}

Tensor::View::View(Tensor& parent)
    : m_parent(parent), m_position({}) {

    m_shape = m_parent.getShape()[m_position];

    size_t numDims = m_shape.getNumDims();
    m_stride.assign(numDims, 1);
}

Tensor::View::View(const Tensor& parent)
    : m_parent((Tensor&)parent), m_position({}) {

    m_shape = m_parent.getShape()[m_position];

    size_t numDims = m_shape.getNumDims();
    m_stride.assign(numDims, 1);
}



Tensor::View Tensor::View::broadcast(Tensor& source, const Tensor::Shape& targetShape) {
    Tensor::Shape srcShape = source.getShape();

    if (srcShape.getNumDims() > targetShape.getNumDims()) {
        throw std::runtime_error("Can't broadcast tensor with more dimensions than target: " +
            srcShape.toString() + " to " + targetShape.toString());
    }

    size_t dimOffset = targetShape.getNumDims() - srcShape.getNumDims();

    // prefix dims (new dims) get stride 0
    std::vector<size_t> stride(targetShape.getNumDims(), 0);

    // suffix dims: match with source
    for (size_t i = 0; i < srcShape.getNumDims(); i++) {
        size_t srcDim = srcShape.getDim(i);
        size_t tgtDim = targetShape.getDim(i + dimOffset);

        if (srcDim == tgtDim) {
            stride[i + dimOffset] = 1;
        } else if (srcDim == 1) {
            // source dim is 1, broadcast to target dim
            stride[i + dimOffset] = 0;
        } else {
            throw std::runtime_error("Can't broadcast shape " + srcShape.toString() +
                " to " + targetShape.toString() + ": dimension " +
                std::to_string(i) + " is " + std::to_string(srcDim) +
                " but target is " + std::to_string(tgtDim));
        }
    }

    // validate prefix-only zeros
    bool seenNonZero = false;
    for (size_t s : stride) {
        if (seenNonZero && s == 0) {
            throw std::runtime_error("Can't broadcast shape " + srcShape.toString() +
                " to " + targetShape.toString() +
                ": would require non-prefix zero strides");
        }
        if (s != 0) seenNonZero = true;
    }

    return Tensor::View(source, stride, targetShape, BroadcastTag{});
}


void Tensor::View::print() const {
    std::cout << "View at position: ";
    for (auto e : m_position) std::cout << e << " ";
    std::cout << "\n";
    m_parent.print(m_position);
}

Tensor& Tensor::View::getParent() const {
    return m_parent;
}

std::vector<size_t> Tensor::View::getPosition() const {
    return m_position;
}

size_t Tensor::View::getOffset() const {
    if (m_position.empty()) {
        return 0;
    }

    Tensor::Shape blockShape = m_shape;
    size_t blockSize = blockShape.getNumElements();

    size_t blockOffset = 1;
    for (size_t d : m_position) {
        blockOffset *= (d + 1);
    }

    return (blockOffset - 1) * blockSize;
}

Tensor::Shape Tensor::View::getShape() const {
    return m_shape;
}

std::vector<size_t> Tensor::View::getStride() const {
    return m_stride;
}

Tensor Tensor::View::copy() const {
    auto shape = getShape();
    auto backend = getParent().getBackend();
    Tensor result(shape, backend);

    // set the whole tensor
    std::vector<size_t> position = {};
    result.set(position, *this);

    return result;
} 


Tensor Tensor::View::operator+(const Tensor& rhs) const {
    Tensor current = copy();
    return current + rhs;
}

Tensor Tensor::View::operator-(const Tensor& rhs) const {
    Tensor current = copy();
    return current - rhs;
}

Tensor Tensor::View::operator*(const Tensor& rhs) const {
    Tensor current = copy();
    return current * rhs;
}

Tensor Tensor::View::operator/(const Tensor& rhs) const {
    Tensor current = copy();
    return current / rhs;
}


Tensor Tensor::View::operator+(const Tensor::View& rhs) const {
    Tensor current = copy();
    return current + rhs;
}

Tensor Tensor::View::operator-(const Tensor::View& rhs) const {
    Tensor current = copy();
    return current - rhs;
}

Tensor Tensor::View::operator*(const Tensor::View& rhs) const {
    Tensor current = copy();
    return current * rhs;
}

Tensor Tensor::View::operator/(const Tensor::View& rhs) const {
    Tensor current = copy();
    return current / rhs;
}


Tensor Tensor::View::operator=(const Tensor& rhs) {
    m_parent.set(m_position, rhs);
    return m_parent;
}

Tensor Tensor::View::operator=(const Tensor::View& rhs) {
    m_parent.set(m_position, rhs);
    return m_parent;
}

Tensor::View Tensor::View::operator[](size_t idx) const {
    std::vector<size_t> position = m_position;
    position.push_back(idx);

    std::vector<size_t> stride;
    for (size_t i = 1; i < m_stride.size(); i++) {
        stride.push_back(m_stride[i]);
    }

    if (stride.empty()) stride.push_back(1);

    // TODO: should get the 
    Tensor::View results(m_parent, position, stride);
    return results;
}

bool Tensor::View::operator==(const Tensor& rhs) const {
    return m_parent.compare(m_position, rhs);
}

bool Tensor::View::operator==(const Tensor::View& rhs) const {
    return m_parent.compare(m_position, rhs);
}

bool Tensor::View::operator!=(const Tensor& rhs) const {
    return !(this->operator==(rhs));
}

bool Tensor::View::operator!=(const Tensor::View& rhs) const {
    return !(this->operator==(rhs));
}