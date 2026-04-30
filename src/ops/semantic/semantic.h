#ifndef SEMANTIC_H
#define SEMANTIC_H

#include "nforge/core/tensor.h"
#include "nforge/core/tensor_shape.h"
#include "nforge/core/tensor_view.h"
#include "nforge/core/tensor_layout.h"

namespace semantic {

struct BinaryOpContext {
    TensorLayout lhs;
    TensorLayout rhs;
    TensorLayout out;
};

struct ReductionContext {
    TensorLayout lhs;
    TensorLayout out;
    TensorLayout block;
};

BinaryOpContext buildContext(const Tensor::View& lhs, const Tensor::View& rhs);
ReductionContext buildReductionContext(const Tensor::View& lhs, size_t dim);

BinaryOpContext validateBinaryOperation(const Tensor::View& lhs, const Tensor::View& rhs);
ReductionContext validateReduction(const Tensor::View& lhs, size_t dim);




}  // namespace semantic

#endif  // SEMANTIC_H