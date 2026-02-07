#include "ops/semantic/semantic.h"


namespace nforge::semantic {

void ensureSameBackend(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.getBackend() != rhs.getBackend()) {
        throw std::runtime_error("Can not perform binary operation on tensor on different devices " 
            + lhs.backendString() + " and " + rhs.backendString() + " of shapes " 
            + lhs.shape().toString() + " and " + rhs.shape().toString());
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
            return cntLhs;
        case ShapeMatch::ScalarRhs:
            return cntRhs;
        case ShapeMatch::EqualCount:
            return cntLhs;
        case ShapeMatch::Incompatible:
            return 0;
        default:
            return 0;
    }
}

BinaryOpContext buildContext(const Tensor::View& lhs, const Tensor::View& rhs) {
    BinaryOpContext ctx;
    ctx.lhsOffset = lhs.getOffset();
    ctx.rhsOffset = rhs.getOffset();
    ctx.shapeMatch = getShapeRelation(lhs.getShape(), rhs.getShape());
    ctx.count = getOperationCount(lhs.getShape(), rhs.getShape(), ctx.shapeMatch);

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
    const Tensor& lhsParent = lhs.getParent();
    const Tensor& rhsParent = rhs.getParent();

    ensureSameBackend(lhsParent, rhsParent);

    return buildContext(lhs, rhs);
}


} // nforge::semantic