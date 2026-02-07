#ifndef TENSOR_VIEW_H
#define TENSOR_VIEW_H

#include "nforge/core/tensor.h"

class Tensor::View {
public:
    View(Tensor& parent, const std::vector<size_t>& index);
    View(const Tensor& parent);

    void print() const;

    // refrenced tensor
    Tensor& getParent() const;

    // position of this view in the tensor it refrences
    std::vector<size_t> getPosition() const;

    // number of elements preceding this view
    size_t getOffset() const;
    
    // shape of the view
    Tensor::Shape getShape() const;
    
    Tensor operator=(const Tensor& other);
    Tensor operator=(const Tensor::View& other);
    Tensor::View operator[](size_t idx) const;

    bool operator==(const Tensor& other) const;
    bool operator==(const Tensor::View& other) const;

    bool operator!=(const Tensor& other) const;
    bool operator!=(const Tensor::View& other) const;
    
private:
    Tensor& m_parent;
    std::vector<size_t> m_position;
};

#endif // TENSOR_VIEW_H