#ifndef TENSOR_VIEW_H
#define TENSOR_VIEW_H

#include "nforge/core/tensor.h"
#include "nforge/core/tensor_layout.h"
#include "nforge/core/tensor_shape.h"

/// Non-owning reference to a sub region of a Tensor.
///
/// A View describes a sub region of a Tensor via an position, shape, and stride layout.
/// Multiple views can reference the same underlying data at different offsets and shapes.
class Tensor::View {
public:
	/// Full view at the origin.
	View(Tensor& parent);

	/// View of a sub tensor starting at `position`.
	View(Tensor& parent, const std::vector<size_t>& position);

	/// View with explicit position and `TensorLayout`.
	View(Tensor& parent, const std::vector<size_t>& position, const TensorLayout& layout);

	/// Implicit conversion from Tensor.
	View(const Tensor& parent);

	/// Creates a broadcast view of `source` to `shape`.
	/// Size-1 dimensions are broadcast by setting their stride to 0.
	static Tensor::View broadcast(Tensor& source, const Tensor::Shape& shape);

	/// Creates a subsampled view of `src`.
	/// A factor of 0 freezes that dimension to a single element (stride 0).
	static Tensor::View subsample(const View& src, const std::vector<size_t>& factors);

	/// Prints the view to stdout.
	void print() const;

	/// Returns the referenced tensor.
	inline Tensor& getParent() const { return m_parent; }

	/// Returns the origin position within the parent tensor.
	inline std::vector<size_t> getPosition() const { return m_position; }

	/// Returns the element offset from the parent's data start.
	inline size_t getOffset() const { return m_layout.offset; }

	/// Returns the shape of the viewed region.
	inline Tensor::Shape getShape() const { return Tensor::Shape(m_layout); }

	/// Returns the logical stride per dimension, normalized by the parent's base stride.
	std::vector<size_t> getStride() const;

	/// Returns the underlying physical layout.
	const TensorLayout& getLayout() const { return m_layout; }

	/// Returns "CPU" or "CUDA".
	std::string getBackendString() const { return m_parent.getBackendString(); }

	/// Returns the backend enum.
	inline Backend getBackend() const { return m_parent.getBackend(); }

	/// Deep copies the viewed region into a new tensor.
	Tensor copy() const;

	/// Copies the viewd elements into a flat vector
	std::vector<float> toVector() const;

	/// Elementwise addition with a tensor or view. Copies then computes.
	Tensor operator+(const Tensor::View& rhs) const;

	/// Elementwise subtraction with a tensor or view. Copies then computes.
	Tensor operator-(const Tensor::View& rhs) const;

	/// Elementwise multiplication with a tensor or view. Copies then computes.
	Tensor operator*(const Tensor::View& rhs) const;

	/// Elementwise division by a tensor or view. Copies then computes.
	Tensor operator/(const Tensor::View& rhs) const;

	/// In-place elementwise addition with a tensor or view. Modifies the parent tensor.
	void operator+=(const Tensor::View& rhs);

	/// In-place elementwise subtraction with a tensor or view. Modifies the parent tensor.
	void operator-=(const Tensor::View& rhs);

	/// In-place elementwise multiplication with a tensor or view. Modifies the parent tensor.
	void operator*=(const Tensor::View& rhs);

	/// In-place elementwise division by a tensor or view. Modifies the parent tensor.
	void operator/=(const Tensor::View& rhs);

	/// Reduces dimensions [dim, rank) by averaging. Result shape is shape[0:dim].
	Tensor mean(size_t dim = 0) const;

	/// Reduces dimensions [dim, rank) by summation. Result shape is shape[0:dim].
	Tensor sum(size_t dim = 0) const;

	/// Reduces dimensions [dim, rank) by taking the minimum. Result shape is shape[0:dim].
	Tensor min(size_t dim = 0) const;

	/// Reduces dimensions [dim, rank) by taking the maximum. Result shape is shape[0:dim].
	Tensor max(size_t dim = 0) const;

	/// Reduces dimensions [dim, rank) by taking the product. Result shape is shape[0:dim].
	Tensor prod(size_t dim = 0) const;

	/// L2 norm (scalar tensor equal to `sqrt(sum(x^2))`).
	Tensor norm() const;

	/// Copies data from a tensor into the referenced position of this view.
	Tensor::View operator=(const Tensor& rhs);

	/// Copies data from a view into the referenced position of this view.
	Tensor::View operator=(const Tensor::View& rhs);

	/// Assigns a pure float to a View.
	/// @note Only works on scalar-shaped views, e.g views with only 1 element.
	Tensor::View operator=(float scalar);

	/// Indexes into the first dimension of this view.
	Tensor::View operator[](size_t idx) const;

	/// Matrix multiplication. Inputs must be 2D or 3D tensors.
	/// 2D: (N, M) @ (M, K) => (N, K).
	///
	/// 3D: (B, N, M) @ (B, M, K) => (B, N, K).
	///
	/// Batch dims must be broadcastable, match or be 1.
	Tensor matmul(const Tensor::View& rhs) const;

	/// Strided subsampling view. Views every `strides[i]`-th element along dim `i`.
	Tensor::View subsample(std::vector<size_t> strides) const;

	/// Returns true if shape and every element matches `rhs`.
	/// @note Exact match, which is unstable for floats. Consider using `.isClose()`
	bool isEqual(const Tensor& rhs) const;

	/// Returns true if shape and every element matches `rhs`.
	/// @note Exact match, which is unstable for floats. Consider using `.isClose()`
	bool isEqual(const Tensor::View& rhs) const;

	/// Returns true if shape or any element does not match `rhs`.
	/// @note Exact match, which is unstable for floats. Consider using `.isClose()`
	bool isNotEqual(const Tensor& rhs) const;

	/// Returns true if shape or any element does not match `rhs`.
	/// @note Exact match, which is unstable for floats. Consider using `.isClose()`
	bool isNotEqual(const Tensor::View& rhs) const;

	/// Elementwise equal. Returns a tensor of 0.0 / 1.0.
	Tensor operator==(const Tensor::View& rhs) const;

	/// Elementwise not equal. Returns a tensor of 0.0 / 1.0.
	Tensor operator!=(const Tensor::View& rhs) const;

	/// Elementwise less than. Returns a tensor of 0.0 / 1.0.
	Tensor operator<(const Tensor::View& rhs) const;

	/// Elementwise less or equal. Returns a tensor of 0.0 / 1.0.
	Tensor operator<=(const Tensor::View& rhs) const;

	/// Elementwise greater than. Returns a tensor of 0.0 / 1.0.
	Tensor operator>(const Tensor::View& rhs) const;

	/// Elementwise greater or equal. Returns a tensor of 0.0 / 1.0.
	Tensor operator>=(const Tensor::View& rhs) const;

	/// Elementwise closeness check within `tolerance`. Returns a tensor of 0.0 / 1.0.
	/// @param tolerance Maximum absolute difference (default: 1e-5).
	Tensor isClose(const Tensor::View& rhs, float tolerance = 1e-5f) const;

private:
	// Differentiates the broadcast constructor from public constructors.
	struct BroadcastTag {};

	// Constructs a view with explicit stride and shape. Used by broadcast().
	View(Tensor& parent, const std::vector<size_t>& stride, const Tensor::Shape& shape,
	     BroadcastTag);

	Tensor& m_parent;
	std::vector<size_t> m_position;
	TensorLayout m_layout;
};

#endif  // TENSOR_VIEW_H
