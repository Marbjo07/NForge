#ifndef SEMANTIC_H
#define SEMANTIC_H

#include "nforge/core/tensor.h"

namespace nforge::semantic {

struct BinaryOpContext {
    size_t lhsOffset = 0;
    size_t rhsOffset = 0;
    size_t count = 0;
};


BinaryOpContext validateBinaryOperation(const Tensor& lhs, const Tensor& rhs);

BinaryOpContext validateBinaryOperation(const Tensor::View& lhs, const Tensor::View& rhs);


} // nforge::semantic


#endif // SEMANTIC_H