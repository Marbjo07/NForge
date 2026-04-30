#include "ops/semantic/semantic.h"

namespace semantic {

void ensureSameBackend(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.getBackend() != rhs.getBackend()) {
        throw std::runtime_error("Can not perform binary operation on tensor on different devices " 
            + lhs.getBackendString() + " and " + rhs.getBackendString() + " of shapes " 
            + lhs.getShape().toString() + " and " + rhs.getShape().toString());
    }
}

static TensorLayout layoutFromView(const Tensor::View& v) {
    return v.getLayout();
}

Tensor::Shape broadcastShapes(const Tensor::Shape& a, const Tensor::Shape& b) {
    const size_t rankA = a.getNumDims();
    const size_t rankB = b.getNumDims();
    const size_t rankOut = std::max(rankA, rankB);

    std::vector<size_t> out(rankOut);

    // a dimension of size 1 can be broadcasted
    for (size_t i = 0; i < rankOut; ++i) {
        // one if dim does not exist
        const size_t dimA = (i < rankA) ? a.getDim(rankA - 1 - i) : 1;
        const size_t dimB = (i < rankB) ? b.getDim(rankB - 1 - i) : 1;

        size_t dim;
        if (dimA == dimB) {
            dim = dimA;
        } 
        else if (dimA == 1) {
            dim = dimB;
        } 
        else if (dimB == 1) {
            dim = dimA;
        } 
        else {
            throw std::runtime_error(
                "Cannot broadcast shapes " + a.toString() +
                " and " + b.toString());
        }

        out[rankOut - 1 - i] = dim;
    }

    return Tensor::Shape(out);
}

static TensorLayout broadcastTo(TensorLayout src, const Tensor::Shape& target) {
    TensorLayout dst{};
    dst.rank   = target.getNumDims();
    dst.offset = src.offset;

    int pad = (int)dst.rank - (int)src.rank;   // align right
    for (int d = 0; d < dst.rank; d++) {
        dst.shape[d] = target.getDim(d);

        int sd = d - pad;

        if (sd < 0 || src.shape[sd] == 1) {
            dst.strides[d] = 0;
        } 
        else {
            dst.strides[d] = src.strides[sd];
        }
    }
    return dst;
}


BinaryOpContext buildContext(const Tensor::View& lhs, const Tensor::View& rhs) {
    Tensor::Shape outShape = broadcastShapes(lhs.getShape(), rhs.getShape());

    BinaryOpContext ctx;
    ctx.lhs = broadcastTo(layoutFromView(lhs), outShape);
    ctx.rhs = broadcastTo(layoutFromView(rhs), outShape);
    ctx.out = outShape.toContiguousLayout();
    return ctx;
}

ReductionContext buildReductionContext(const Tensor::View& lhs, size_t dim) {
    Tensor::Shape lhsShape = lhs.getShape();

    Tensor::Shape blockShape = lhsShape.getSlice(dim, lhsShape.getNumDims());
    Tensor::Shape outShape = lhsShape.getSlice(0, dim);

    ReductionContext ctx;
    ctx.lhs = lhsShape;
    ctx.out = outShape;
    ctx.block = blockShape;
    return ctx;
}

IndexContext buildIndexContext(const Tensor::View& src, size_t idx) {
    TensorLayout layout = src.getLayout();
    size_t offset = layout.offset + layout.strides[0] * idx;
    auto shape = Tensor::Shape(layout)[0];
    
    // ensure rank > 0
    size_t newRank = std::max((int)layout.rank - 1, 1);

    // shift by one, removing leading dim
    std::vector<size_t> strides(newRank, 1);
    for (int d = 0; d < (int)layout.rank - 1; d++) {
        strides[d] = layout.strides[d + 1];
    }

    TensorLayout out(shape, strides, offset);
    
    IndexContext res{out};
    return res;
}



BinaryOpContext validateBinaryOperation(const Tensor::View& lhs, const Tensor::View& rhs) {
    ensureSameBackend(lhs.getParent(), rhs.getParent());

    return buildContext(lhs, rhs);
}

ReductionContext validateReduction(const Tensor::View& lhs, size_t dim) {
    if (dim > lhs.getShape().getNumDims()) {
        throw std::runtime_error("Can not reduce Tensor of shape " 
            + lhs.getShape().toString() + " with along dim " + std::to_string(dim));
    }

    return buildReductionContext(lhs, dim);
}

IndexContext validateIndexing(const Tensor::View& src, size_t idx) {
    Tensor::Shape shape = src.getShape();
    if (idx < 0 || idx >= shape.getDim(0)) {
        throw std::out_of_range("Index " + std::to_string(idx) 
            + " is out of bounds. Tensor view shape: "
            + shape.toString());
    }

    return buildIndexContext(src, idx);
}



}  // namespace semantic