#include "src/ops/semantic/semantic.h"


namespace nforge::semantic {

    
BinaryOpContext validateBinaryOperation(const Tensor& lhs, const Tensor& rhs) {
    // must have equal backend
    if (lhs.getBackend() != rhs.getBackend()) {
        throw std::runtime_error("Can not perform binary operation on tensor on different devices " 
            + lhs.backendString() + " and " + rhs.backendString() + " of shapes " 
            + lhs.shape().toString() + " and " + rhs.shape().toString());
    }

    // must have equal shapes
    if (lhs.shape() != rhs.shape()) {
        throw std::runtime_error("Can not perform binary operations on tensors of different shapes, "
            + lhs.shape().toString() + " and " + rhs.shape().toString());
    }

    Tensor::Shape shape = rhs.shape();
    
    BinaryOpContext ctx;
    ctx.lhsOffset = 0;
    ctx.rhsOffset = 0;
    ctx.count = shape.getNumElements();

    return ctx;
}


BinaryOpContext validateBinaryOperation(const Tensor::View& lhs, const Tensor::View& rhs) {
    const Tensor& lhsParent = lhs.getParent();
    const Tensor& rhsParent = rhs.getParent();
    
    // must have equal backend
    if (lhsParent.getBackend() != rhsParent.getBackend()) {
        throw std::runtime_error("Can not perform binary operation on tensor views on different devices " 
            + lhsParent.backendString() + " and " + rhsParent.backendString() + " of shapes " 
            + lhs.getShape().toString() + " and " + rhs.getShape().toString());
    }


    // must have equal shapes
    if (lhs.getShape() != rhs.getShape()) {
        throw std::runtime_error("Can not perform binary operations on tensors of different shapes, "
            + lhs.getShape().toString() + " and " + rhs.getShape().toString());
    }

    Tensor::Shape shape = rhs.getShape();

    BinaryOpContext ctx;
    ctx.lhsOffset = lhs.getOffset();
    ctx.rhsOffset = rhs.getOffset();
    ctx.count = shape.getNumElements();

    return ctx;
}


} // nforge::semantic