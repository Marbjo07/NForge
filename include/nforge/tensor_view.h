#ifndef TENSOR_VIEW_H
#define TENSOR_VIEW_H

#include "nforge/tensor.h"

class Tensor::View {
public:
    View(Tensor& parent, const std::vector<size_t>& index);

    void print() const;

    Tensor operator=(const Tensor& other);

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