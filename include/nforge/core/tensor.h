#ifndef TENSOR_H
#define TENSOR_H

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/// Available backends for tensor storage and operations.
enum class Backend { CPU, CUDA };

/// Multi-dimensional array with backend specific implementation.
///
/// A `Tensor` owns a contiguous block of float data managed by a backend specific `Impl` object.
/// Elements can be accessed through `Tensor::View`, which describes a sub region via offset, shape,
/// and stride layout.
class Tensor {
public:
	class Impl;
	class CPUImpl;
	class CUDAImpl;

	class View;
	class Shape;

public:
	/// Constructs a tensor with the given shape, zero-initialized.
	Tensor(const Tensor::Shape& shape, Backend backend = Backend::CPU);

	/// Constructs a tensor with the given shape, zero-initialized.
	Tensor(const std::initializer_list<size_t>& shape, Backend backend = Backend::CPU);

	/// Constructs a tensor and fills every element with `value`.
	Tensor(const Tensor::Shape& shape, float value, Backend backend = Backend::CPU);

	/// Constructs a tensor and fills every element with `value`.
	Tensor(const std::initializer_list<size_t>& shape, float value, Backend backend = Backend::CPU);

	/// Constructs a scalar tensor, shape (1).
	Tensor(float value, Backend backend = Backend::CPU);

	/// Copy constructor. Performs a deep copy.
	Tensor(const Tensor& tensor);

	/// Constructs a tensor by taking ownership of a backend implementation.
	/// @param impl  Backend implementation, ownership is transferred.
	Tensor(std::unique_ptr<Tensor::Impl> impl, Backend backend = Backend::CPU);

	/// Destructor.
	~Tensor();

	/// Transfers data to a different backend. No-op if already on that backend.
	void to(Backend newBackend);

	/// Fills all elements with `value`.
	void fillAll(float value);

	/// Fills all elements with random values in [-1, 1].
	void fillRand();

	/// Prints the tensor to stdout.
	void print() const;

	/// Prints the block starting at `position` to stdout.
	void print(const std::vector<size_t>& position) const;

	/// Returns the tensor shape.
	Tensor::Shape getShape() const;

	/// Returns "CPU" or "CUDA".
	std::string getBackendString() const;

	/// Returns the backend enum.
	inline Backend getBackend() const { return m_backend; }

	/// Returns a string representation of the underlying data.
	std::string getDataString() const;

	/// Returns the total number of elements.
	size_t getNumElements() const;

	/// Copies all elements into a flat vector (row-major order).
	std::vector<float> toVector() const;

	/// Replaces the block starting at `position` with the data from `rhs`.
	void set(const std::vector<size_t>& position, const Tensor::View& rhs);

	/// Returns true if shape and every element matches `rhs`.
	bool compare(const Tensor::View& rhs) const;

	/// Returns true if the block at `position` matches `rhs`.
	bool compare(const std::vector<size_t>& position, const Tensor::View& rhs) const;

	/// Elementwise addition with a tensor or view.
	Tensor operator+(const Tensor::View& rhs) const;

	/// Elementwise subtraction with a tensor or view.
	Tensor operator-(const Tensor::View& rhs) const;

	/// Elementwise multiplication with a tensor or view.
	Tensor operator*(const Tensor::View& rhs) const;

	/// Elementwise division by a tensor or view.
	Tensor operator/(const Tensor::View& rhs) const;

	/// Elementwise addition with a pure float.
	Tensor operator+(float scalar) const;

	/// Elementwise subtraction with a pure float.
	Tensor operator-(float scalar) const;

	/// Elementwise multiplication with a pure float.
	Tensor operator*(float scalar) const;

	/// Elementwise division by a pure float.
	Tensor operator/(float scalar) const;

	/// Elementwise addition of a pure float and a tensor.
	friend Tensor operator+(float scalar, const Tensor& rhs);

	/// Elementwise subtraction of a tensor from a pure float.
	friend Tensor operator-(float scalar, const Tensor& rhs);

	/// Elementwise multiplication of a pure float and a tensor.
	friend Tensor operator*(float scalar, const Tensor& rhs);

	/// Elementwise division of a pure float by a tensor.
	friend Tensor operator/(float scalar, const Tensor& rhs);

	/// In-place elementwise addition with a tensor or view.
	void operator+=(const Tensor::View& rhs);

	/// In-place elementwise subtraction with a tensor or view.
	void operator-=(const Tensor::View& rhs);

	/// In-place elementwise multiplication with a tensor or view.
	void operator*=(const Tensor::View& rhs);

	/// In-place elementwise division by a tensor or view.
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

	/// Matrix multiplication. Inputs must be 2D or 3D tensors.
	/// 2D: (N, M) @ (M, K) => (N, K).
	/// 3D: (B, N, M) @ (B, M, K) => (B, N, K).
	/// Batch dims must be broadcastable, match or be 1.
	Tensor matmul(const Tensor::View& rhs) const;

	/// Indexes into the first dimension, returning a view of the sub-tensor.
	Tensor::View operator[](size_t idx) const;

	/// Strided sub-sampling view. Views every `strides[i]`-th element along dim `i`.
	/// @param strides Length must match rank of tensor or be scalar 0.
	Tensor::View subsample(std::vector<size_t> strides) const;

	/// Copies data from another tensor.
	Tensor& operator=(const Tensor& rhs);

	/// Copies data from a view.
	Tensor& operator=(const Tensor::View& rhs);

	/// Assigns `scalar` to every element.
	Tensor& operator=(float scalar);

	/// Returns true if all elements equal those in `rhs`.
	/// @note Exact match, which is unstable for floats. Consider using `.isClose()`
	bool operator==(const Tensor::View& rhs) const;

	/// Returns true if any element differs from `rhs`.
	/// @note Exact match, which is unstable for floats. Consider using `.isClose()`
	bool operator!=(const Tensor::View& rhs) const;

	/// Elementwise less than. Returns a tensor of 0.0 / 1.0.
	Tensor operator<(const Tensor::View& rhs) const;

	/// Elementwise less or equal. Returns a tensor of 0.0 / 1.0.
	Tensor operator<=(const Tensor::View& rhs) const;

	/// Elementwise greater than. Returns a tensor of 0.0 / 1.0.
	Tensor operator>(const Tensor::View& rhs) const;

	/// Elementwise greater or equal. Returns a tensor of 0.0 / 1.0.
	Tensor operator>=(const Tensor::View& rhs) const;

	/// Elementwise closeness check within `tolerance`. Returns a tensor of 0.0 / 1.0.
	/// @param tolerance  Maximum absolute or relative difference (default: 1e-5).
	Tensor isClose(const Tensor::View& rhs, float tolerance = 1e-5f) const;

private:
	Backend m_backend;
	std::unique_ptr<Impl> m_impl;

	/// Applies `op` element-wise via Impl after broadcasting. Returns a new tensor.
	/// @tparam BinaryOp  Member function pointer on Impl, e.g. `&Impl::add`.
	template <typename BinaryOp>
	Tensor applyBinaryOp(const Tensor::View& rhs, BinaryOp op) const;

	/// Applies `op` in-place via Impl. The output layout must match `*this`.
	/// @tparam BinaryOp  Member function pointer on Impl, e.g. `&Impl::iadd`.
	template <typename BinaryOp>
	void applyInplaceBinaryOp(const Tensor::View& rhs, BinaryOp op);

	/// Applies reduction `op` along dimensions [dim, rank) via Impl.
	/// @tparam ReductionOp  Member function pointer on Impl, e.g. `&Impl::sum`.
	template <typename ReductionOp>
	Tensor applyReduction(size_t dim, ReductionOp op) const;
};

#endif  // TENSOR_H