#include "ops/semantic/semantic.h"

namespace nforge::semantic {

void ensureSameBackend(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.getBackend() != rhs.getBackend()) {
        throw std::runtime_error("Can not perform binary operation on tensor on different devices " 
            + lhs.getBackendString() + " and " + rhs.getBackendString() + " of shapes " 
            + lhs.getShape().toString() + " and " + rhs.getShape().toString());
    }
}

void ensureSameShape(Tensor::Shape lhsShape, Tensor::Shape rhsShape) {
    if (lhsShape != rhsShape) {
        throw std::runtime_error("Can not perform binary operations on tensors of different shapes, " 
            + lhsShape.toString() + " and " + rhsShape.toString());
    }
}

ShapeMatch getShapeRelation(const Tensor::Shape& lhs, const Tensor::Shape& rhs) {
    // keep order, must match order of ShapeMatch
    if (lhs == rhs) {
        return ShapeMatch::Equal;
    }

    if (lhs.isScalar()) {
        return ShapeMatch::ScalarLhs;
    }

    if (rhs.isScalar()) {
        return ShapeMatch::ScalarRhs;
    }

    if (lhs.getNumElements() == lhs.getNumElements()) {
        return ShapeMatch::EqualCount;
    }

    return ShapeMatch::Incompatible;
}

size_t getOperationCount(const Tensor::Shape& lhs, const Tensor::Shape& rhs, ShapeMatch shapeMatch) {
    size_t cntLhs = lhs.getNumElements();
    size_t cntRhs = rhs.getNumElements();

    switch (shapeMatch) {
        case ShapeMatch::Equal:
            return cntLhs;
        case ShapeMatch::ScalarLhs:
            return cntRhs;
        case ShapeMatch::ScalarRhs:
            return cntLhs;
        case ShapeMatch::EqualCount:
            return cntLhs;
        case ShapeMatch::Incompatible:
            return 0;
        default:
            return 0;
    }
}

BinaryOpContext buildContext(const Tensor::View& lhs, const Tensor::View& rhs) {
    Tensor::Shape lhsShape = lhs.getShape();
    Tensor::Shape rhsShape = rhs.getShape();

    BinaryOpContext ctx;
    ctx.lhsOffset = lhs.getOffset();
    ctx.rhsOffset = rhs.getOffset();
    ctx.shapeMatch = getShapeRelation(lhsShape, rhsShape);
    ctx.count = getOperationCount(lhsShape, rhsShape, ctx.shapeMatch);

    return ctx;
}

BinaryOpContext validateBinaryOperation(const Tensor& lhs, const Tensor& rhs) {
    return validateBinaryOperation(Tensor::View(lhs), Tensor::View(rhs));
}

BinaryOpContext validateBinaryOperation(const Tensor& lhs, const Tensor::View& rhs) {
    return validateBinaryOperation(Tensor::View(lhs), rhs);
}

BinaryOpContext validateBinaryOperation(const Tensor::View& lhs, const Tensor& rhs) {
    return validateBinaryOperation(lhs, Tensor::View(rhs));
}

BinaryOpContext validateBinaryOperation(const Tensor::View& lhs, const Tensor::View& rhs) {
    ensureSameBackend(lhs.getParent(), rhs.getParent());

    return buildContext(lhs, rhs);
}

}  // namespace nforge::semantic