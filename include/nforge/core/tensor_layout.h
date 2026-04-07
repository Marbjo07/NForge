#include <stddef.h>

#ifndef TENSOR_LAYOUT_H
#define TENSOR_LAYOUT_H

#define MAX_DIMS 8

struct TensorLayout {
    size_t shape[MAX_DIMS];
    size_t strides[MAX_DIMS];
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