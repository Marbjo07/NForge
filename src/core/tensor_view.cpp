#include "nforge/core/tensor_view.h"

Tensor::View::View(Tensor& parent, const std::vector<size_t>& index)
    : m_parent(parent), m_position(index), m_stride(index.size(), 1) {
    m_shape = m_parent.getShape()[m_position];
}

Tensor::View::View(Tensor& parent, const std::vector<size_t>& index, const std::vector<size_t>& stride, const Tensor::Shape& shape)
    : m_parent(parent), m_position(index), m_stride(stride) {
        
    Tensor::Shape blockShape = m_parent.getShape()[m_position];
    
    if (m_stride.size() != blockShape.getNumDims()) {
        throw std::runtime_error("Stride must have same number of dimensions as indexed block");
    }
    if (m_stride.size() != shape.getNumDims()) {
        throw std::runtime_error("Stride must have same number of dimensions as view shape");
    }

    // only valid with a prefix of zeros
    bool seenNonZero = false;
    for (size_t s : m_stride) {
        if (seenNonZero && s == 0) {
            throw std::runtime_error("Zero stride only allowed as a prefix");
        }
        if (s != 0) seenNonZero = true;
    }

    // validate that the target shape is possible
    for (size_t i = 0; i < m_stride.size(); i++) {
        // the target can be anything 
        if (m_stride[i] == 0) continue;


        size_t blockDim = blockShape.getDim(i);
        if (shape.getDim(i) != blockDim / m_stride[i]) {
            throw std::runtime_error("Stride and block shape does not result in target shape");
        }
    }

    m_shape = shape;
}


Tensor::View::View(Tensor& parent, const std::vector<size_t>& index, const std::vector<size_t>& stride)
    : m_parent(parent), m_position(index), m_stride(stride) {
        
    Tensor::Shape blockShape = m_parent.getShape()[m_position];

    if (m_stride.size() != blockShape.getNumDims()) {
        throw std::runtime_error("Stride must have same number of dimensions as indexed block");
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

Tensor::View Tensor::View::broadcast(Tensor& source, const Tensor::Shape& shape) {
    Tensor::Shape srcShape = source.getShape();

    if (srcShape.getNumDims() > shape.getNumDims()) {
        throw std::runtime_error("Can't broadcast tensor with more dimensions than target!" + 
            srcShape.toString() + " to " + shape.toString());
    }

    // check if any dims are larger than target shape
    // the new dims are added at the front, the tensor is stacked
    size_t dimOffset = shape.getNumDims() - srcShape.getNumDims();

    std::vector<size_t> stride(shape.getNumDims(), 0);
    
    for (size_t i = 0; i < srcShape.getNumDims(); i++) {
        size_t srcDim = srcShape.getDim(i);
        size_t targetDim = shape.getDim(i + dimOffset);

        // check if broadcastable
        if (targetDim % srcDim != 0) {
            throw std::runtime_error("Can't broadcast tensor of shape" + srcShape.toString() +
            " to tensor of shape " + shape.toString());
        }

        stride[i + dimOffset] = targetDim / srcDim;
    }

    
    return Tensor::View(source, {}, stride, shape);
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

    Tensor::View results(m_parent, position);
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