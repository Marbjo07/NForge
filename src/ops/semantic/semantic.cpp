#include "ops/semantic/semantic.h"

namespace semantic {

void ensureSameBackend(const Tensor::View& lhs, const Tensor::View& rhs) {
	if (lhs.getParent().getBackend() != rhs.getParent().getBackend()) {
		const auto& lhsBackendStr = lhs.getParent().getBackendString();
		const auto& rhsBackendStr = rhs.getParent().getBackendString();

		const auto& lhsShapeStr = lhs.getShape().toString();
		const auto& rhsShapeStr = rhs.getShape().toString();

		throw std::runtime_error(
		    "Can not perform binary operation on tensor on different devices " + lhsBackendStr +
		    " and " + rhsBackendStr + " of shapes " + lhsShapeStr + " and " + rhsShapeStr);
	}
}

inline Tensor::Shape broadcastShapes(const Tensor::Shape& lhs, const Tensor::Shape& rhs) {
	size_t rankLhs = lhs.getNumDims();
	size_t rankRhs = rhs.getNumDims();
	size_t rankOut = std::max(rankLhs, rankRhs);

	std::vector<size_t> outDims(rankOut);

	for (size_t i = 1; i <= rankOut; i++) {
		// pad with 1s
		size_t dimLhs = (i <= rankLhs) ? lhs.getDim(rankLhs - i) : 1;
		size_t dimRhs = (i <= rankRhs) ? rhs.getDim(rankRhs - i) : 1;

		// dimensions must match or be 1 to broadcast
		if (dimLhs != dimRhs && dimLhs != 1 && dimRhs != 1) {
			throw std::runtime_error("Can not broadcast shapes " + lhs.toString() + " and " +
			                         rhs.toString());
		}

		// output is the non 1 dimension or 1 if both are 1
		outDims[rankOut - i] = (dimLhs == 1) ? dimRhs : dimLhs;
	}

	return Tensor::Shape(outDims);
}

inline TensorLayout broadcastTo(const TensorLayout& src, const Tensor::Shape& target) {
	size_t targetRank = target.getNumDims();
	std::vector<size_t> strides(targetRank, 0);

	int pad = (int)targetRank - (int)src.rank;  // align right
	for (int d = pad; d < targetRank; d++) {
		// use stride if dim is not size 1
		if (src.shape[d - pad] != 1) {
			strides[d] = src.strides[d - pad];
		}
	}

	return TensorLayout(target, strides, src.offset);
}

BinaryOpContext BinaryOpContext::build(const Tensor::View& lhs, const Tensor::View& rhs) {
	ensureSameBackend(lhs, rhs);

	const Tensor::Shape& outShape = broadcastShapes(lhs.getShape(), rhs.getShape());

	BinaryOpContext ctx;
	ctx.lhs = broadcastTo(lhs.getLayout(), outShape);
	ctx.rhs = broadcastTo(rhs.getLayout(), outShape);
	ctx.out = outShape.toContiguousLayout();
	return ctx;
}

ReductionContext ReductionContext::build(const Tensor::View& lhs, size_t dim) {
	if (dim > lhs.getShape().getNumDims()) {
		throw std::runtime_error("Can not reduce Tensor of shape " + lhs.getShape().toString() +
		                         " with along dim " + std::to_string(dim));
	}

	Tensor::Shape& lhsShape = lhs.getShape();
	Tensor::Shape& outShape = lhsShape.getSlice(0, dim);
	Tensor::Shape& blockShape = lhsShape.getSlice(dim, lhsShape.getNumDims());

	ReductionContext ctx;
	ctx.lhs = lhsShape;
	ctx.out = outShape;
	ctx.block = blockShape;
	return ctx;
}

IndexContext IndexContext::build(const Tensor::View& src, size_t idx) {
	const Tensor::Shape& srcShape = src.getShape();
	if (idx < 0 || idx >= srcShape.getDim(0)) {
		throw std::out_of_range("Index " + std::to_string(idx) +
		                        " is out of bounds. Tensor view shape: " + srcShape.toString());
	}

	TensorLayout layout = src.getLayout();
	size_t offset = layout.offset + layout.strides[0] * idx;
	const Tensor::Shape& shape = Tensor::Shape(layout)[0];

	// ensure rank > 0
	size_t newRank = std::max((int)layout.rank - 1, 1);

	// shift by one, removing leading dim
	std::vector<size_t> strides(newRank, 1);
	for (int d = 0; d < (int)layout.rank - 1; d++) {
		strides[d] = layout.strides[d + 1];
	}

	TensorLayout out(shape, strides, offset);

	IndexContext ctx;
	ctx.out = out;
	return ctx;
}


inline void validateRanksMatmul(size_t lhsRank, size_t rhsRank) {
	if (lhsRank < 2 || lhsRank > 3 || rhsRank < 2 || rhsRank > 3) {
		throw std::runtime_error("matmul: inputs must be 2D or 3D tensors");
	}
}

inline void validateInnerDimsMatmul(const Tensor::Shape& lhsShape, const Tensor::Shape& rhsShape,
                                    size_t lhsRank, size_t rhsRank) {
	if (lhsShape.getDim(lhsRank - 1) != rhsShape.getDim(rhsRank - 2)) {
		throw std::runtime_error("matmul: inner dimensions must match, got " + lhsShape.toString() +
		                         " and " + rhsShape.toString());
	}
}

inline size_t computeBatchMatmul(const Tensor::Shape& lhsShape, const Tensor::Shape& rhsShape,
                                 size_t lhsRank, size_t rhsRank) {
	const size_t lhsBatch = (lhsRank == 3) ? lhsShape.getDim(0) : 1;
	const size_t rhsBatch = (rhsRank == 3) ? rhsShape.getDim(0) : 1;

	if (lhsRank == 3 && rhsRank == 3 && lhsBatch != rhsBatch && lhsBatch != 1 && rhsBatch != 1) {
		throw std::runtime_error("matmul: batch dimensions must match or be 1, got " +
		                         lhsShape.toString() + " and " + rhsShape.toString());
	}
	return std::max(lhsBatch, rhsBatch);
}

inline TensorLayout computeOutputLayoutMatmul(size_t batch, size_t m, size_t p) {
	std::vector<size_t> outDims;
	if (batch > 1) {
		outDims.push_back(batch);
	}
	outDims.push_back(m);
	outDims.push_back(p);

	return Tensor::Shape(outDims).toContiguousLayout();
}


MatmulContext MatmulContext::build(const Tensor::View& lhs, const Tensor::View& rhs) {
	ensureSameBackend(lhs, rhs);

	const Tensor::Shape& lhsShape = lhs.getShape();
	const Tensor::Shape& rhsShape = rhs.getShape();
	size_t lhsRank = lhsShape.getNumDims();
	size_t rhsRank = rhsShape.getNumDims();

	validateRanksMatmul(lhsRank, rhsRank);
	validateInnerDimsMatmul(lhsShape, rhsShape, lhsRank, rhsRank);

	MatmulContext ctx;
	ctx.m = lhsShape.getDim(lhsRank - 2);
	ctx.k = lhsShape.getDim(lhsRank - 1);
	ctx.p = rhsShape.getDim(rhsRank - 1);
	ctx.batch = computeBatchMatmul(lhsShape, rhsShape, lhsRank, rhsRank);

	ctx.lhs = lhs.getLayout();
	ctx.rhs = rhs.getLayout();
	ctx.out = computeOutputLayoutMatmul(ctx.batch, ctx.m, ctx.p);

	return ctx;
}


InplaceBinaryOpContext InplaceBinaryOpContext::build(const Tensor::View& lhs,
                                                     const Tensor::View& rhs) {
	const BinaryOpContext& ctx = BinaryOpContext::build(lhs, rhs);
	const TensorLayout& lhsLayout = lhs.getLayout();

	if (lhsLayout != ctx.lhs) {
		throw std::runtime_error(
		    "Can not apply in-place operator where infered output is different from lhs!");
	}

	InplaceBinaryOpContext res;
	res.lhs = ctx.lhs;
	res.rhs = ctx.rhs;

	return res;
}


}  // namespace semantic