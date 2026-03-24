#include "nforge/core/tensor_view.h"

Tensor::View::View(Tensor& parent, const std::vector<size_t>& index)
    : m_parent(parent), m_position(index), m_stride(index.size(), 1) {
    m_shape = m_parent.getShape()[m_position];
}

Tensor::View::View(Tensor& parent, const std::vector<size_t>& index, const std::vector<size_t>& stride)
    : m_parent(parent), m_position(index), m_stride(stride) {
        
    Tensor::Shape blockShape = m_parent.getShape()[m_position];
    
    if (m_stride.size() != blockShape.getNumDims()) {
        throw std::runtime_error("Stride must have same number of dimensions as shape");
    }

    std::vector<size_t> shape(m_stride.size());
   
    for (size_t i = 0; i < m_stride.size(); i++) {
        size_t parentDim = blockShape.getDim(i);

        if (m_stride[i] == 0) {
            shape[i] = 1;
        } else {
            shape[i] = parentDim / m_stride[i];
        }
    }

    m_shape = Tensor::Shape(shape);
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

    
    return Tensor::View(source, {}, stride);
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