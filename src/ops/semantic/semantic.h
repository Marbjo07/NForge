#ifndef SEMANTIC_H
#define SEMANTIC_H

#include "nforge/core/tensor.h"
#include "nforge/core/tensor_layout.h"
#include "nforge/core/tensor_shape.h"
#include "nforge/core/tensor_view.h"

namespace semantic {

namespace detail {
// forces all contexts to use build method.
class OperationContext {
protected:
	OperationContext() = default;
};
}  // namespace detail


class BinaryOpContext : detail::OperationContext {
public:
	TensorLayout lhs;
	TensorLayout rhs;
	TensorLayout out;

	static BinaryOpContext build(const Tensor::View& lhs, const Tensor::View& rhs);
};

class InplaceBinaryOpContext : detail::OperationContext {
public:
	TensorLayout lhs;
	TensorLayout rhs;

	static InplaceBinaryOpContext build(const Tensor::View& lhs, const Tensor::View& rhs);
};


class ReductionContext : detail::OperationContext {
public:
	TensorLayout lhs;
	TensorLayout out;
	TensorLayout block;

	static ReductionContext build(const Tensor::View& lhs, size_t dim);
};


class IndexContext : detail::OperationContext {
public:
	TensorLayout out;

	static IndexContext build(const Tensor::View& src, size_t idx);
};


class MatmulContext : detail::OperationContext {
public:
	TensorLayout lhs;
	TensorLayout rhs;
	TensorLayout out;
	size_t batch;
	size_t m;
	size_t k;
	size_t p;

	static MatmulContext build(const Tensor::View& lhs, const Tensor::View& rhs);
};

}  // namespace semantic

#endif  // SEMANTIC_H