#ifndef SEMANTIC_H
#define SEMANTIC_H

#include "nforge/core/tensor.h"

namespace semantic {

// sorted by precedence, examples show shape array
enum class ShapeMatch {
    Equal,        // [2,3,4] vs [2,3,4]
    ScalarLhs,    // [1] vs [2,3,4]
    ScalarRhs,    // [2,3,4] vs [1]
    EqualCount,   // [12] vs [3,4] (same total elements, different shape)
    Incompatible  // no other match
};

struct BinaryOpContext {
    size_t lhsOffset = 0;
    size_t rhsOffset = 0;
    size_t count = 0;
    ShapeMatch shapeMatch = ShapeMatch::Incompatible;
};

ShapeMatch getShapeRelation(const Tensor::Shape& lhs, const Tensor::Shape& rhs);

BinaryOpContext validateBinaryOperation(const Tensor& lhs, const Tensor& rhs);
BinaryOpContext validateBinaryOperation(const Tensor& lhs, const Tensor::View& rhs);
BinaryOpContext validateBinaryOperation(const Tensor::View& lhs, const Tensor& rhs);
BinaryOpContext validateBinaryOperation(const Tensor::View& lhs, const Tensor::View& rhs);

BinaryOpContext buildContext(const Tensor::View& lhs, const Tensor::View& rhs);

}  // namespace semantic

#endif  // SEMANTIC_H