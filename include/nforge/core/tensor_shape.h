#ifndef TENSOR_SHAPE_H
#define TENSOR_SHAPE_H

#include <string>
#include <vector>

#include "nforge/core/tensor.h"

struct TensorLayout;

/// Describes the extent of each dimension of a Tensor.
///
/// Trailing ones are stripped during equality comparison,
/// so `{3, 4, 1} == {3, 4}`.
class Tensor::Shape {
public:
	/// Default constructor. Creates an empty (0-dim) shape.
	Shape() = default;

	/// From a vector of dimension sizes. Empty vector becomes {1}.
	Shape(const std::vector<size_t>& dims);

	/// From an initializer list. Empty list becomes {1}.
	Shape(const std::initializer_list<size_t>& dims);

	/// From a layout's active rank dimensions. Empty layout becomes {1}.
	Shape(const TensorLayout& layout);

	/// Equality, ignoring trailing ones. So `{3, 4, 1} == {3, 4}`.
	bool operator==(const Tensor::Shape& other) const;

	/// Negation of operator==.
	bool operator!=(const Tensor::Shape& other) const;

	/// Returns the number of dimensions.
	size_t getNumDims() const;

	/// Returns the product of all dimension sizes.
	size_t getNumElements() const;

	/// Returns the extent of dimension `idx`.
	size_t getDim(size_t idx) const;

	/// True if the shape is {1}.
	bool isScalar() const;

	/// Shape of the sub-tensor after indexing the first dimension.
	/// Returns {1} if already 1D.
	Tensor::Shape operator[](size_t index) const;

	/// Shape of the sub-tensor after indexing with a multi-dim position.
	/// Returns {1} if no dimensions remain.
	Tensor::Shape operator[](const std::vector<size_t>& position) const;

	/// Removes the first dimension. Throws if the shape is empty.
	Tensor::Shape removeLeadingDimension() const;

	/// Returns shape with dimensions in [start, end).
	Tensor::Shape getSlice(size_t start, size_t end) const;

	/// Returns a string like "{ 3 4 5 }".
	std::string toString() const;

	/// Returns the dimension sizes as a vector.
	std::vector<size_t> toVector() const;

	/// Strips trailing 1s, always keeping at least one dimension.
	std::vector<size_t> withoutTrailingOnes() const;

	/// Creates a row-major contiguous layout (last dim stride 1).
	TensorLayout toContiguousLayout() const;

	/// Returns the row-major strides as a vector.
	std::vector<size_t> getContiguousStrides() const;

private:
	std::vector<size_t> m_dimensions;
};

#endif  // TENSOR_SHAPE_H
