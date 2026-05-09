#include "nforge/core/tensor_layout.h"

TensorLayout::TensorLayout(const Tensor::Shape& _shape)
    : rank(_shape.getNumDims()) {
    assert(_shape.getNumDims() <= MAX_DIMS);
        
    auto dims = _shape.toVector(); 
    std::copy(dims.begin(), dims.end(), shape.begin());
    
    auto contigStrides = _shape.getContiguousStrides();
    std::copy(contigStrides.begin(), contigStrides.end(), strides.begin());
}

TensorLayout::TensorLayout(const Tensor::Shape& _shape, const std::vector<size_t>& _strides) 
    : rank(_shape.getNumDims()) {
    assert(_shape.getNumDims() <= MAX_DIMS);
    assert(_strides.size() == _shape.getNumDims());

    auto dims = _shape.toVector(); 
    std::copy(dims.begin(), dims.end(), shape.begin());
    
    std::copy(_strides.begin(), _strides.end(), strides.begin());
}

TensorLayout::TensorLayout(const Tensor::Shape& _shape, const std::vector<size_t>& _strides, size_t _offset) 
    : rank(_shape.getNumDims()), offset(_offset) {
    assert(_shape.getNumDims() <= MAX_DIMS);
    assert(_strides.size() == _shape.getNumDims());
    
    auto dims = _shape.toVector(); 
    std::copy(dims.begin(), dims.end(), shape.begin());
    
    std::copy(_strides.begin(), _strides.end(), strides.begin());
}

bool TensorLayout::operator==(const TensorLayout& rhs) const {
    if (rank != rhs.rank) return false;
    if (offset != rhs.offset) return false;

    for (int d = rank - 1; d >= 0; d--) {
        if (shape[d] != rhs.shape[d]) return false;
        if (strides[d] != rhs.strides[d]) return false;
    }

    return true;
}

bool TensorLayout::operator!=(const TensorLayout& rhs) const {
    return !operator==(rhs);
}