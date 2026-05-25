#ifndef SEMANTIC_H
#define SEMANTIC_H

#include "nforge/core/tensor.h"
#include "nforge/core/tensor_layout.h"
#include "nforge/core/tensor_shape.h"
#include "nforge/core/tensor_view.h"

namespace semantic {

struct BinaryOpContext {
	static BinaryOpContext build(const Tensor::View& lhs, const Tensor::View& rhs);

	TensorLayout lhs;
	TensorLayout rhs;
	TensorLayout out;

private:
	BinaryOpContext() = default;  // enforce use of build
};

struct InplaceBinaryOpContext {
	static InplaceBinaryOpContext build(const Tensor::View& lhs, const Tensor::View& rhs);

	TensorLayout lhs;
	TensorLayout rhs;

private:
	InplaceBinaryOpContext() = default;  // enforce use of build
};


struct ReductionContext {
	static ReductionContext build(const Tensor::View& lhs, size_t dim);

	TensorLayout lhs;
	TensorLayout out;
	TensorLayout block;

private:
	ReductionContext() = default;  // enforce use of build
};


struct IndexContext {
	static IndexContext build(const Tensor::View& src, size_t idx);

	TensorLayout out;

private:
	IndexContext() = default;  // enforce use of build
};


struct MatmulContext {
	static MatmulContext build(const Tensor::View& lhs, const Tensor::View& rhs);

	TensorLayout lhs;
	TensorLayout rhs;
	TensorLayout out;
	size_t batch;
	size_t m;
	size_t k;
	size_t p;

private:
	MatmulContext() = default;  // enforce use of build
};

}  // namespace semantic

#endif  // SEMANTIC_H