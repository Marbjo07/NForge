#include "nforge/core/tensor_shape.h"

#include <numeric>
#include <algorithm>
#include <stdexcept>

Tensor::Shape::Shape(const std::vector<size_t>& dims) : m_dimensions(dims) {
    if (dims.empty()) {
        m_dimensions.push_back(1); // scalar tensors have shape {1}
    }
}

Tensor::Shape::Shape(const std::initializer_list<size_t>& dims) : m_dimensions(dims) {
    if (dims.size() == 0) {
        m_dimensions.push_back(1); // scalar tensors have shape {1}
    }
}

bool Tensor::Shape::operator==(const Shape &other) const {
    return this->withoutTrailingOnes() == other.withoutTrailingOnes();
}

bool Tensor::Shape::operator!=(const Shape &other) const {
    return !(this->operator==(other));
}

size_t Tensor::Shape::getNumDims() const { 
    return m_dimensions.size();
}

Tensor::Shape Tensor::Shape::operator[](size_t index) const { 
    if (m_dimensions.size() == 1) {
        return Tensor::Shape({1});
    }

    std::vector<size_t> dims(m_dimensions.begin() + 1, m_dimensions.end());
    return Tensor::Shape(dims);
}

Tensor::Shape Tensor::Shape::operator[](const std::vector<size_t>& position) const { 
    size_t numIndexedDims = position.size();
    size_t remainDims = m_dimensions.size() - numIndexedDims;
    if (remainDims <= 0) {
        return Tensor::Shape({1});
    }

    std::vector<size_t> dims(m_dimensions.begin() + numIndexedDims, m_dimensions.end());
    return Tensor::Shape(dims);
}

size_t Tensor::Shape::getNumElements() const {
    if (m_dimensions.empty()) {
        return 0;
    }

    return std::accumulate(m_dimensions.begin(), m_dimensions.end(),
                           size_t(1), std::multiplies<size_t>());
}

size_t Tensor::Shape::getDim(size_t idx) const {
    return m_dimensions[idx];
}

bool Tensor::Shape::isScalar() const {
    return getNumElements() == 1;
}

Tensor::Shape Tensor::Shape::removeLeadingDimension() const {
    if (m_dimensions.empty()) {
        throw std::runtime_error("Cannot remove dimension from empty shape");
    }
    return Shape(std::vector<size_t>(m_dimensions.begin() + 1, m_dimensions.end()));
}

Tensor::Shape Tensor::Shape::getSlice(size_t start, size_t end) const {
    if (start > end || end > m_dimensions.size()) {
        throw std::out_of_range("Invalid slice range");
    }
    return Shape(std::vector<size_t>(m_dimensions.begin() + start, m_dimensions.begin() + end));
}

std::string Tensor::Shape::toString() const {
    std::string out = "{ ";
    for (size_t dim : m_dimensions) {
        out += std::to_string(dim) + " ";
    }
    out += "}";
    return out;
}

std::vector<size_t> Tensor::Shape::toVector() const {
    return m_dimensions;
}

std::vector<size_t> Tensor::Shape::withoutTrailingOnes() const {
    std::vector<size_t> result = m_dimensions;
    while (!result.empty() && result.back() == 1) {
        result.pop_back();
    }
    // maintain at least one dimension
    if (result.empty()) {
        result.push_back(1); 
    }
    return result;
}