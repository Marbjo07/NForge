#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include "nforge/core/tensor.h"
#include "nforge/core/tensor_layout.h"

/// Abstract interface for backend specific tensor storage and operations.
///
/// All data access goes through TensorLayout descriptors, not raw indices.
///
/// The caller must ensure layouts are valid, since the implementation trusts that `rhsLayout` is
/// correct for `rhsImpl` and `lhsLayout` is correct for its own memory.
///
/// The caller is responsible for broadcasting.
class Tensor::Impl {
public:
	Impl() = default;
	virtual ~Impl() = default;

	/// Fills all elements with `value`.
	virtual void fillAll(float value) = 0;

	/// Fills all elements with random values in [-1, 1].
	virtual void fillRand() = 0;

	/// Prints the entire tensor to stdout.
	virtual void print() const = 0;

	/// Prints the block starting at `position` to stdout.
	virtual void print(const std::vector<size_t>& position) const = 0;

	/// Returns the total number of elements.
	virtual size_t getNumElements() const = 0;

	/// Returns the tensor shape.
	virtual Tensor::Shape getShape() const = 0;

	/// Copies all elements into a flat vector (row-major order).
	virtual std::vector<float> toVector() const = 0;

	/// Returns a string representation of the data.
	virtual std::string toString() const = 0;

	/// Deep copies this implementation.
	virtual std::unique_ptr<Tensor::Impl> clone() const = 0;

	/// Copies data from a host float array into this backend's storage.
	/// @param data  Source array (must have at least `count` elements).
	/// @param count  Number of elements to copy.
	virtual void copyFromHost(const float* data, size_t count) = 0;

	/// Copies data from `rhsImpl` with `rhsLayout` into `this` with `lhsLayout`.
	virtual void set(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	                 const TensorLayout& rhsLayout) = 0;

	/// Returns true if the data with `lhsLayout` matches `rhsImpl` with `rhsLayout`.
	virtual bool compare(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	                     const TensorLayout& rhsLayout) const = 0;

	/// Elementwise addition. Returns a new Impl with the result with `outLayout`.
	virtual std::unique_ptr<Tensor::Impl> add(const TensorLayout& lhsLayout,
	                                          const Tensor::Impl* rhsImpl,
	                                          const TensorLayout& rhsLayout,
	                                          const TensorLayout& outLayout) const = 0;

	/// Elementwise subtraction. Returns a new Impl with the result with `outLayout`.
	virtual std::unique_ptr<Tensor::Impl> sub(const TensorLayout& lhsLayout,
	                                          const Tensor::Impl* rhsImpl,
	                                          const TensorLayout& rhsLayout,
	                                          const TensorLayout& outLayout) const = 0;

	/// Elementwise multiplication. Returns a new Impl with the result with `outLayout`.
	virtual std::unique_ptr<Tensor::Impl> mul(const TensorLayout& lhsLayout,
	                                          const Tensor::Impl* rhsImpl,
	                                          const TensorLayout& rhsLayout,
	                                          const TensorLayout& outLayout) const = 0;

	/// Elementwise division. Returns a new Impl with the result with `outLayout`.
	virtual std::unique_ptr<Tensor::Impl> div(const TensorLayout& lhsLayout,
	                                          const Tensor::Impl* rhsImpl,
	                                          const TensorLayout& rhsLayout,
	                                          const TensorLayout& outLayout) const = 0;

	/// In-place elementwise addition. Modifies `lhsLayout` in place.
	virtual void iadd(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	                  const TensorLayout& rhsLayout) = 0;

	/// In-place elementwise subtraction. Modifies `lhsLayout` in place.
	virtual void isub(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	                  const TensorLayout& rhsLayout) = 0;

	/// In-place elementwise multiplication. Modifies `lhsLayout` in place.
	virtual void imul(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	                  const TensorLayout& rhsLayout) = 0;

	/// In-place elementwise division. Modifies `lhsLayout` in place.
	virtual void idiv(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	                  const TensorLayout& rhsLayout) = 0;

	/// Reduces dimensions [dim, rank) by summation. Output with `outLayout`.
	virtual std::unique_ptr<Tensor::Impl> sum(const TensorLayout& layout,
	                                          const TensorLayout& blockLayout,
	                                          const TensorLayout& outLayout) const = 0;

	/// Reduces dimensions [dim, rank) by taking the minimum. Output with `outLayout`.
	virtual std::unique_ptr<Tensor::Impl> min(const TensorLayout& layout,
	                                          const TensorLayout& blockLayout,
	                                          const TensorLayout& outLayout) const = 0;

	/// Reduces dimensions [dim, rank) by taking the maximum. Output with `outLayout`.
	virtual std::unique_ptr<Tensor::Impl> max(const TensorLayout& layout,
	                                          const TensorLayout& blockLayout,
	                                          const TensorLayout& outLayout) const = 0;

	/// Reduces dimensions [dim, rank) by taking the product. Output with `outLayout`.
	virtual std::unique_ptr<Tensor::Impl> prod(const TensorLayout& layout,
	                                           const TensorLayout& blockLayout,
	                                           const TensorLayout& outLayout) const = 0;

	/// L2 norm of the tensor described by `layout`.
	virtual std::unique_ptr<Tensor::Impl> norm(const TensorLayout& layout) const = 0;

	/// For each block, tests whether all element evaluate to True (non-zero).
	/// Reduces dimensions [dim, rank) by applying logical AND.
	/// Returns a tensor of 0.0 / 1.0 with `outLayout`.
	virtual std::unique_ptr<Tensor::Impl> all(const TensorLayout& layout,
	                                          const TensorLayout& blockLayout,
	                                          const TensorLayout& outLayout) const = 0;

	/// For each block, tests whether any element evaluate to True (non-zero).
	/// Reduces dimensions [dim, rank) by applying logical OR.
	/// Returns a tensor of 0.0 / 1.0 with `outLayout`.
	virtual std::unique_ptr<Tensor::Impl> any(const TensorLayout& layout,
	                                          const TensorLayout& blockLayout,
	                                          const TensorLayout& outLayout) const = 0;

	/// Matrix multiplication. The last two dims of each layout are the matrix dims.
	/// `batch`, `m`, `k`, `p` describe the decomposition of the matmul problem.
	///
	/// 2D: (m, k) @ (k, p) => (m, p).
	///
	/// 3D: (batch, m, k) @ (batch, k, p) => (batch, m, p).
	virtual std::unique_ptr<Tensor::Impl> matmul(const TensorLayout& lhsLayout,
	                                             const Tensor::Impl* rhsImpl,
	                                             const TensorLayout& rhsLayout,
	                                             const TensorLayout& outLayout, size_t batch,
	                                             size_t m, size_t k, size_t p) const = 0;


	/// Elementwise less than. Returns a tensor of 0.0 / 1.0 with `outLayout`.
	virtual std::unique_ptr<Tensor::Impl> less(const TensorLayout& lhsLayout,
	                                           const Tensor::Impl* rhsImpl,
	                                           const TensorLayout& rhsLayout,
	                                           const TensorLayout& outLayout) const = 0;

	/// Elementwise less or equal. Returns a tensor of 0.0 / 1.0 with `outLayout`.
	virtual std::unique_ptr<Tensor::Impl> lessEqual(const TensorLayout& lhsLayout,
	                                                const Tensor::Impl* rhsImpl,
	                                                const TensorLayout& rhsLayout,
	                                                const TensorLayout& outLayout) const = 0;

	/// Elementwise greater than. Returns a tensor of 0.0 / 1.0 with `outLayout`.
	virtual std::unique_ptr<Tensor::Impl> greater(const TensorLayout& lhsLayout,
	                                              const Tensor::Impl* rhsImpl,
	                                              const TensorLayout& rhsLayout,
	                                              const TensorLayout& outLayout) const = 0;

	/// Elementwise greater or equal. Returns a tensor of 0.0 / 1.0 with `outLayout`.
	virtual std::unique_ptr<Tensor::Impl> greaterEqual(const TensorLayout& lhsLayout,
	                                                   const Tensor::Impl* rhsImpl,
	                                                   const TensorLayout& rhsLayout,
	                                                   const TensorLayout& outLayout) const = 0;

	/// Elementwise closeness within `tolerance`. Returns a tensor of 0.0 / 1.0 with `outLayout`.
	virtual std::unique_ptr<Tensor::Impl> isClose(const TensorLayout& lhsLayout,
	                                              const Tensor::Impl* rhsImpl,
	                                              const TensorLayout& rhsLayout,
	                                              const TensorLayout& outLayout,
	                                              float tolerance) const = 0;
};

#endif  // TENSOR_IMPL_H