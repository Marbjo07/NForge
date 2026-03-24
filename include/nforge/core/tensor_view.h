#ifndef TENSOR_VIEW_H
#define TENSOR_VIEW_H

#include "nforge/core/tensor.h"
#include "nforge/core/tensor_shape.h"

class Tensor::View {
   public:
    View(Tensor& parent, const std::vector<size_t>& index);
    View(Tensor& parent, const std::vector<size_t>& index, const std::vector<size_t>& stride);
    View(Tensor& parent);

    // implicit casting
    View(const Tensor& parent);

    static Tensor::View broadcast(Tensor& source, const Tensor::Shape& shape);

    void print() const;

    // refrenced tensor
    Tensor& getParent() const;

    // position of this view in the tensor it refrences
    std::vector<size_t> getPosition() const;

    // number of elements preceding this view
    size_t getOffset() const;

    // shape of the view
    Tensor::Shape getShape() const;

    std::vector<size_t> getStride() const;

    // creates a copy of the viewed tensor
    Tensor copy() const;

    Tensor operator+(const Tensor& rhs) const;
    Tensor operator-(const Tensor& rhs) const;
    Tensor operator*(const Tensor& rhs) const;
    Tensor operator/(const Tensor& rhs) const;

    Tensor operator+(const Tensor::View& rhs) const;
    Tensor operator-(const Tensor::View& rhs) const;
    Tensor operator*(const Tensor::View& rhs) const;
    Tensor operator/(const Tensor::View& rhs) const;

    Tensor operator=(const Tensor& rhs);
    Tensor operator=(const Tensor::View& rhs);
    Tensor::View operator[](size_t idx) const;

    bool operator==(const Tensor& rhs) const;
    bool operator==(const Tensor::View& rhs) const;

    bool operator!=(const Tensor& rhs) const;
    bool operator!=(const Tensor::View& rhs) const;

   private:
    Tensor& m_parent;
    std::vector<size_t> m_position, m_stride;
    Tensor::Shape m_shape;
};

#endif  // TENSOR_VIEW_H