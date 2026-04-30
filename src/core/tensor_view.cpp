#include "nforge/core/tensor_view.h"

#include "backend/tensor_impl.h"
#include "ops/semantic/semantic.h"

Tensor::View::View(Tensor& parent)
    : m_parent(parent), m_layout(parent.getShape()), m_position({}) {}

    
Tensor::View::View(const Tensor& parent)
    // const correctness ¯\_(ツ)_/¯, dont know him
    : m_parent((Tensor&)parent), m_layout(parent.getShape()), m_position({}) {}


Tensor::View::View(Tensor& parent, const std::vector<size_t>& position)
    : m_parent(parent), m_layout(parent.getShape()[position]), m_position(position) {
    auto parentStrides = parent.getShape().getContiguousStrides();
    size_t offset = 0;
    for (size_t d = 0; d < position.size(); d++) {
        offset += position[d] * parentStrides[d];
    }
    m_layout.offset = offset;
}


Tensor::View::View(Tensor& parent, const std::vector<size_t>& stride,
                   const Tensor::Shape& shape, BroadcastTag)
    : m_parent(parent), m_layout(shape, stride), m_position({}) {}
      

Tensor::View::View(Tensor& parent, const std::vector<size_t>& position,
                   const TensorLayout& layout) 
    : m_parent(parent), m_layout(layout), m_position(position) {}

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
    return m_layout.offset;
}

Tensor::Shape Tensor::View::getShape() const {
    return Tensor::Shape(m_layout);
}

const TensorLayout& Tensor::View::getLayout() const {
    return m_layout;
}

std::vector<size_t> Tensor::View::getStride() const {
    std::vector<size_t> stride(m_layout.strides.begin(), m_layout.strides.begin() + m_layout.rank);
    std::vector<size_t> baseStride = m_parent.getShape().getContiguousStrides();

    size_t dimOffset = m_position.size();
    for (size_t d = 0; d < stride.size(); d++) {
        size_t baseDim = d + dimOffset;
        if (baseDim < baseStride.size() && baseStride[baseDim] > 0) {
            stride[d] = stride[d] / baseStride[baseDim];
        }
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


Tensor::View Tensor::View::operator=(const Tensor& rhs) {
    Tensor::View rhsView(rhs);
    
    auto ctx = semantic::validateBinaryOperation(*this, rhsView);
    if (Tensor::Shape(ctx.out) != getShape()) {
        throw std::invalid_argument("set(): rhs shape does not broadcast to target shape");
    }

    m_parent.m_impl->set(ctx.lhs, rhs.m_impl.get(), ctx.rhs);
    return *this;
}

Tensor::View Tensor::View::operator=(const Tensor::View& rhs) {
    auto ctx = semantic::validateBinaryOperation(*this, rhs);
    if (Tensor::Shape(ctx.out) != getShape()) {
        throw std::invalid_argument("set(): rhs shape does not broadcast to target shape");
    }
    
    m_parent.m_impl->set(ctx.lhs, rhs.m_parent.m_impl.get(), ctx.rhs);
    return *this;
}

Tensor::View Tensor::View::operator[](size_t idx) const {
    auto ctx = semantic::validateIndexing(*this, idx);

    std::vector<size_t> position = m_position;
    position.push_back(idx);

    Tensor::View out(m_parent, position, ctx.out);
    return out;
}

Tensor::View Tensor::View::subsample(const Tensor::View& src, const std::vector<size_t>& factors) {

    if (src.getShape().getNumDims() != factors.size()) {
        throw std::runtime_error("Can't subsample view of shape"
            + src.getShape().toString() + " with factors of rank "
            + std::to_string(factors.size()));
    }

    std::vector<size_t> dims = src.getShape().toVector();
    std::vector<size_t> strides(src.m_layout.strides.begin(),
                                src.m_layout.strides.begin() + factors.size());

    for (size_t d = 0; d < factors.size(); d++) {
        size_t factor = factors[d];

        if (factor == 0) {
            dims[d] = 1;
            strides[d] = 0;
        } 
        else {
            dims[d] /= factor;
            strides[d] *= factor;
        }
    }

    Tensor::Shape shape = Tensor::Shape(dims);
    TensorLayout layout(shape, strides, src.m_layout.offset);

    Tensor::View out(src.m_parent, src.m_position, layout);

    return out;
}

Tensor::View Tensor::View::subsample(std::vector<size_t> strides) const {
    size_t rank = m_layout.rank;
    if (strides.size() != rank) {
        if (strides.size() == 1 && strides[0] == 0) {
            strides.assign(rank, 0);
        } 
        else {
            throw std::runtime_error("Can't subsample view with different rank strides than shape.");
        }
    }

    return Tensor::View::subsample(*this, strides);
}


bool Tensor::View::operator==(const Tensor& rhs) const {
    Tensor::View rhsView(rhs);
    if (getShape() != rhsView.getShape()) return false;
    auto ctx = semantic::validateBinaryOperation(*this, rhsView);
    return m_parent.m_impl->compare(ctx.lhs, rhs.m_impl.get(), ctx.rhs);
}

bool Tensor::View::operator==(const Tensor::View& rhs) const {
    if (getShape() != rhs.getShape()) return false;
    auto ctx = semantic::validateBinaryOperation(*this, rhs);
    return m_parent.m_impl->compare(ctx.lhs, rhs.m_parent.m_impl.get(), ctx.rhs);
}

bool Tensor::View::operator!=(const Tensor& rhs) const {
    return !(this->operator==(rhs));
}

bool Tensor::View::operator!=(const Tensor::View& rhs) const {
    return !(this->operator==(rhs));
}