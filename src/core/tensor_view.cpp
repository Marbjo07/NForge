#include "nforge/core/tensor_view.h"

Tensor::View::View(Tensor& parent, const std::vector<size_t>& index)
    : m_parent(parent), m_position(index) {
}

Tensor::View::View(const Tensor& parent)
    : m_parent((Tensor&)parent), m_position({}) {
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

    Tensor::Shape blockShape = getShape();
    size_t blockSize = blockShape.getNumElements();

    size_t blockOffset = 1;
    for (size_t d : m_position) {
        blockOffset *= (d + 1);
    }

    return (blockOffset - 1) * blockSize;
}

Tensor::Shape Tensor::View::getShape() const {
    return m_parent.getShape()[m_position];
}

Tensor Tensor::View::operator=(const Tensor& lhs) {
    m_parent.set(m_position, lhs);
    return m_parent;
}

Tensor Tensor::View::operator=(const Tensor::View& lhs) {
    m_parent.set(m_position, lhs);
    return m_parent;
}

Tensor::View Tensor::View::operator[](size_t idx) const {
    std::vector<size_t> position = m_position;
    position.push_back(idx);

    Tensor::View results(m_parent, position);
    return results;
}

bool Tensor::View::operator==(const Tensor& lhs) const {
    return m_parent.compare(m_position, lhs);
}

bool Tensor::View::operator==(const Tensor::View& lhs) const {
    return m_parent.compare(m_position, lhs);
}

bool Tensor::View::operator!=(const Tensor& lhs) const {
    return !(this->operator==(lhs));
}

bool Tensor::View::operator!=(const Tensor::View& lhs) const {
    return !(this->operator==(lhs));
}