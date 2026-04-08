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

BinaryOpContext validateBinaryOperation(const Tensor::View& lhs, const Tensor::View& rhs);

BinaryOpContext buildContext(const Tensor::View& lhs, const Tensor::View& rhs);

}  // namespace semantic

#endif  // SEMANTIC_H