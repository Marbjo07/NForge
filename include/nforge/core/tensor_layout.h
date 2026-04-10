#include <stddef.h>
#include <array>

#include "nforge/core/tensor_shape.h"

#ifndef TENSOR_LAYOUT_H
#define TENSOR_LAYOUT_H

#define MAX_DIMS 8

struct TensorLayout {
    std::array<size_t, MAX_DIMS> shape;
    std::array<size_t, MAX_DIMS> strides;
    size_t offset = 0;
    size_t rank = 0;
    
    TensorLayout() {};
    TensorLayout(const Tensor::Shape& _shape);
    TensorLayout(const Tensor::Shape& _shape, const std::vector<size_t>& _strides);
    TensorLayout(const Tensor::Shape& _shape, const std::vector<size_t>& _strides, size_t _offset);

    TensorLayout(std::array<size_t, MAX_DIMS> shape, 
                 std::array<size_t, MAX_DIMS> strides,
                 size_t offset, size_t rank)
        : shape(shape), strides(strides), offset(offset), rank(rank) {}
};



static inline size_t physicalOffset(size_t linear, const TensorLayout& L) {
    size_t off = L.offset;
    for (int d = L.rank - 1; d >= 0; d--) {
        size_t coord = linear % L.shape[d];
        linear /= L.shape[d];
        off += coord * L.strides[d];
    }
    return off;
}


#endif // TENSOR_LAYOUT_H 