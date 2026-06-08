#include <stddef.h>

#include <array>

#include "nforge/core/tensor_shape.h"

#ifndef TENSOR_LAYOUT_H
#define TENSOR_LAYOUT_H

#define MAX_DIMS 8

/// Describes the memory layout of a tensor (shape, strides, offset, rank).
///
/// Only the first `rank` entries of `shape` and `strides` are active.
/// `0 <= rank && rank <= MAX_DIMS`.
struct TensorLayout {
	/// Shape extent per dimension. Only first `rank` entries active.
	std::array<size_t, MAX_DIMS> shape;

	/// Stride (element count) per dimension. Only first `rank` entries active.
	std::array<size_t, MAX_DIMS> strides;

	/// Storage offset before indexing begins.
	size_t offset = 0;

	/// Number of active dimensions.
	size_t rank = 0;

	/// Default constructor. Rank and offset are zero.
	TensorLayout() : shape{}, strides{} {}

	/// Contiguous (row-major) layout from a shape.
	/// @pre `_shape.getNumDims() <= MAX_DIMS`
	TensorLayout(const Tensor::Shape& _shape);

	/// Layout from shape and explicit strides.
	/// @pre `_shape.getNumDims() <= MAX_DIMS`
	/// @pre `_strides.size() == _shape.getNumDims()`
	TensorLayout(const Tensor::Shape& _shape, const std::vector<size_t>& _strides);

	/// Layout from shape, explicit strides, and storage offset.
	/// @pre `_shape.getNumDims() <= MAX_DIMS`
	/// @pre `_strides.size() == _shape.getNumDims()`
	TensorLayout(const Tensor::Shape& _shape, const std::vector<size_t>& _strides, size_t _offset);

	/// Layout from raw arrays and rank.
	/// @pre `_rank <= MAX_DIMS`
	TensorLayout(std::array<size_t, MAX_DIMS> _shape, std::array<size_t, MAX_DIMS> _strides,
	             size_t _offset, size_t _rank);

	/// True if layouts have equal rank, offset, and active shape/stride entries.
	bool operator==(const TensorLayout& rhs) const;

	/// Negation of operator==.
	bool operator!=(const TensorLayout& rhs) const;
};

/// Converts a linear index to a strided memory offset.
static inline size_t physicalOffset(size_t linear, const TensorLayout& L) {
	size_t off = L.offset;
	for (int d = L.rank - 1; d >= 0; d--) {
		std::lldiv_t results = std::lldiv((long long)linear, (long long)L.shape[d]);

		linear = results.quot;
		off += results.rem * L.strides[d];
	}
	return off;
}

#endif  // TENSOR_LAYOUT_H