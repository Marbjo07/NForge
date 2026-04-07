#include <stddef.h>
#include <array>

#ifndef TENSOR_LAYOUT_H
#define TENSOR_LAYOUT_H

#define MAX_DIMS 8

struct TensorLayout {
    std::array<size_t, MAX_DIMS> shape;
    std::array<size_t, MAX_DIMS> strides;
    size_t offset;
    size_t rank;
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