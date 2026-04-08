#ifndef TENSOR_VIEW_H
#define TENSOR_VIEW_H

#include "nforge/core/tensor.h"
#include "nforge/core/tensor_shape.h"

class Tensor::View {
   public:
    View(Tensor& parent);
    View(Tensor& parent, const std::vector<size_t>& position);

    // implicit casting
    View(const Tensor& parent);

    static Tensor::View broadcast(Tensor& source, const Tensor::Shape& shape);
    static Tensor::View subsample(const View& src, const std::vector<size_t>& factors);

    void print() const;

    // refrenced tensor
    Tensor& getParent() const;

    // position of this view in the tensor it refrences
    std::vector<size_t> getPosition() const;

    // number of elements preceding this view
    size_t getOffset() const;

    // shape of the view
    Tensor::Shape getShape() const;

    // returns the stride for each dim, not the underlying stride
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

    Tensor::View subsample(std::vector<size_t> strides) const;

    bool operator==(const Tensor& rhs) const;
    bool operator==(const Tensor::View& rhs) const;

    bool operator!=(const Tensor& rhs) const;
    bool operator!=(const Tensor::View& rhs) const;

   private:
    // resolves ambiguous overload with initializer list
    struct BroadcastTag {};
    // trusted, used by broadcast.
    View(Tensor& parent, const std::vector<size_t>& stride, const Tensor::Shape& shape, BroadcastTag);
    
    Tensor& m_parent;
    Tensor::Shape m_shape;
    std::vector<size_t> m_stride, m_position;
    size_t m_offset;
};

#endif  // TENSOR_VIEW_H