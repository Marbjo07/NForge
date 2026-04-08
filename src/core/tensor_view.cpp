#include "nforge/core/tensor_view.h"

Tensor::View::View(Tensor& parent)
    : m_parent(parent), m_shape(parent.getShape()), m_offset(0), m_position({}) {
    m_stride = m_shape.getContiguousStrides();
}

Tensor::View::View(const Tensor& parent)
    // const correctness ¯\_(ツ)_/¯, dont know him
    : m_parent((Tensor&)parent), m_shape(parent.getShape()), m_offset(0), m_position({}) {
    m_stride = m_shape.getContiguousStrides();
}

Tensor::View::View(Tensor& parent, const std::vector<size_t>& position)
    : m_parent(parent), m_position(position) {

    auto parentStride = parent.getShape().getContiguousStrides();

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

    auto srcStride = srcShape.getContiguousStrides();
    size_t dimOffset = targetShape.getNumDims() - srcShape.getNumDims();

    // prefix dims (new leading dims) get stride 0
    // regardless of coordinate along that axis
    std::vector<size_t> stride(targetShape.getNumDims(), 0);

    for (size_t i = 0; i < srcShape.getNumDims(); i++) {
        size_t srcDim = srcShape.getDim(i);
        size_t tgtDim = targetShape.getDim(i + dimOffset);

        if (srcDim == tgtDim) {
            stride[i + dimOffset] = srcStride[i];
        } 
        else if (srcDim == 1) {
            // size-1 dim stretched to tgtDim: stride 0 pins it to the single element
            stride[i + dimOffset] = 0;
        } 
        else {
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
    std::vector<size_t> stride(m_stride.size());
    std::vector<size_t> baseStride = m_parent.getShape().getContiguousStrides();
    
    for (size_t d = 0; d < stride.size(); d++) {
        stride[d] = m_stride[d] / baseStride[d];
    }

    return stride;
}

Tensor Tensor::View::copy() const {
    auto shape = getShape();
    auto backend = getParent().getBackend();
    Tensor result(shape, backend);

    
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

Tensor::View Tensor::View::subsample(const Tensor::View& src, const std::vector<size_t>& factors) {
    Tensor::View out = src;  // copy layout
    std::vector<size_t> dims = out.m_shape.toVector(); 
    for (size_t d = 0; d < factors.size(); d++) {
        size_t factor = factors[d];

        if (factor == 0) {
            throw std::runtime_error(
                "Zero stride for dimension" + std::to_string(d));
        }

        // shrink logical shape
        dims[d] /= factor;
        
        // stretch physical stride
        out.m_stride[d] *= factor;
    }

    out.m_shape = Tensor::Shape(dims);

    return out;
}

Tensor::View Tensor::View::subsample(std::vector<size_t> strides) const {
    size_t rank = m_shape.getNumDims();
    if (strides.size() != rank) {
        throw std::runtime_error("Can't subsample view with different rank strides than shape.");
    }

    return Tensor::View::subsample(*this, strides);
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