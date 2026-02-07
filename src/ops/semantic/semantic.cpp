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
    ensureSameShape(lhs.getShape(), rhs.getShape());

    Tensor::Shape shape = rhs.getShape();

    BinaryOpContext ctx;
    ctx.lhsOffset = lhs.getOffset();
    ctx.rhsOffset = rhs.getOffset();
    ctx.count = shape.getNumElements();

    return ctx;
}

} // nforge::semantic