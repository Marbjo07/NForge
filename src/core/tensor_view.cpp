#include "nforge/core/tensor_view.h"


std::vector<size_t> contiguousStridesFor(const Tensor::Shape& shape) {
    std::vector<size_t> stride(shape.getNumDims(), 1);
    
    size_t prod = 1;
    for (int i = (int)shape.getNumDims() - 1; i >= 0; i--) {
        stride[i] = prod;
        prod *= shape.getDim(i);
    }

    return stride;
}

Tensor::View::View(Tensor& parent)
    : m_parent(parent), m_shape(parent.getShape()), m_offset(0), m_position({}) {
    m_stride = contiguousStridesFor(m_shape);
}

Tensor::View::View(const Tensor& parent)
    // const correctness ¯\_(ツ)_/¯, dont know him
    : m_parent((Tensor&)parent), m_shape(parent.getShape()), m_offset(0), m_position({}) {
    m_stride = contiguousStridesFor(m_shape);
}

Tensor::View::View(Tensor& parent, const std::vector<size_t>& position)
    : m_parent(parent), m_position(position) {

    auto parentStride = contiguousStridesFor(parent.getShape());

    m_offset = 0;
    for (size_t d = 0; d < position.size(); d++) {
        m_offset += position[d] * parentStride[d];
    }

    // index away the first dims based on position
    m_shape = parent.getShape()[position];
    m_stride.assign(parentStride.begin() + position.size(), parentStride.end());

    while (m_stride.size() < m_shape.getNumDims()) {
        m_stride.push_back(1);
    }
}


Tensor::View::View(Tensor& parent, const std::vector<size_t>& position,
                   const std::vector<size_t>& stride)
    : m_parent(parent), m_position(position), m_stride(stride) {

    auto parentStride = contiguousStridesFor(parent.getShape());

    m_offset = 0;
    for (size_t d = 0; d < position.size(); d++) {
        m_offset += position[d] * parentStride[d];
    }
    
    m_shape = parent.getShape()[position];
}

Tensor::View::View(Tensor& parent, const std::vector<size_t>& stride,
                   const Tensor::Shape& shape, BroadcastTag)
    : m_parent(parent), m_shape(shape), m_stride(stride),
      m_offset(0), m_position({}) {}
      

Tensor::View Tensor::View::broadcast(Tensor& source, const Tensor::Shape& targetShape) {
    Tensor::Shape srcShape = source.getShape();

    if (srcShape.getNumDims() > targetShape.getNumDims()) {
        throw std::runtime_error("Can't broadcast tensor with more dimensions than target: " +
            srcShape.toString() + " to " + targetShape.toString());
    }

    auto srcStride = contiguousStridesFor(srcShape);
    size_t dimOffset = targetShape.getNumDims() - srcShape.getNumDims();

    // prefix dims (new leading dims) get stride 0
    // regardless of coordinate along that axis
    std::vector<size_t> stride(targetShape.getNumDims(), 0);

    for (size_t i = 0; i < srcShape.getNumDims(); i++) {
        size_t srcDim = srcShape.getDim(i);
        size_t tgtDim = targetShape.getDim(i + dimOffset);

        if (srcDim == tgtDim) {
            stride[i + dimOffset] = srcStride[i];
        } else if (srcDim == 1) {
            // size-1 dim stretched to tgtDim: stride 0 pins it to the single element
            stride[i + dimOffset] = 0;
        } else {
            throw std::runtime_error("Can't broadcast shape " + srcShape.toString() +
                " to " + targetShape.toString() + ": dimension " +
                std::to_string(i) + " is " + std::to_string(srcDim) +
                " but target is " + std::to_string(tgtDim));
        }
    }

    return Tensor::View(source, stride, targetShape, BroadcastTag{});
}


void Tensor::View::print() const {
    auto position = getPosition();
    
    std::cout << "View at position: ";
    for (auto e : position) std::cout << e << " ";
    std::cout << "\n";

    m_parent.print(position);
}

Tensor& Tensor::View::getParent() const {
    return m_parent;
}

std::vector<size_t> Tensor::View::getPosition() const {
    return m_position;
}

size_t Tensor::View::getOffset() const {
    return m_offset;
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
    View out = *this;

    out.m_offset += m_stride[0] * idx;
    out.m_shape = m_shape[0];
    out.m_stride.erase(out.m_stride.begin());
    out.m_position.push_back(idx);

    if (out.m_stride.empty()) {
        out.m_stride.push_back(1);
    }

    return out;
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