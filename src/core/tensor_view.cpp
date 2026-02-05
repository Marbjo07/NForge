#include "nforge/core/tensor_view.h"

Tensor::View::View(Tensor& parent, const std::vector<size_t>& index) 
    : m_parent(parent), m_position(index) {
}

void Tensor::View::print() const {
    m_parent.print(m_position);
}

Tensor& Tensor::View::getParent() const {
    return m_parent;
}

std::vector<size_t> Tensor::View::getPosition() const {
    return m_position;
}

size_t Tensor::View::getOffset() const {
    Tensor::Shape blockShape = getShape();
    size_t blockSize = blockShape.getNumElements();

    size_t blockOffset = 1;
    for (size_t d : m_position) {
        blockOffset *= d;
    }

    return blockOffset * blockSize;
}

Tensor::Shape Tensor::View::getShape() const {
    return m_parent.shape()[m_position];
}

Tensor Tensor::View::operator=(const Tensor& other) {
    m_parent.set(m_position, other);
    return m_parent;
}

Tensor Tensor::View::operator=(const Tensor::View& other) {
    m_parent.set(m_position, other);
    return m_parent;
}

Tensor::View Tensor::View::operator[](size_t idx) const {
    std::vector<size_t> position = m_position;
    position.push_back(idx);

    Tensor::View results(m_parent, position);
    return results;
}

bool Tensor::View::operator==(const Tensor& other) const {
    return m_parent.compare(m_position, other);
}

bool Tensor::View::operator==(const Tensor::View& other) const {
    return m_parent.compare(m_position, other);
}

bool Tensor::View::operator!=(const Tensor& other) const {
    return !(this->operator==(other));
}

bool Tensor::View::operator!=(const Tensor::View& other) const {
    return !(this->operator==(other));
}