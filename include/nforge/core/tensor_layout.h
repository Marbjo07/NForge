#include <stddef.h>

#include <array>

#include "nforge/core/tensor_shape.h"

#ifndef TENSOR_LAYOUT_H
#define TENSOR_LAYOUT_H

#define MAX_DIMS 8

/**
 * @class TensorLayout
 * @brief Describes the memory layout of a tensor.
 *
 * A tensor layout stores:
 * - the tensor shape,
 * - the stride for each dimension,
 * - an optional storage offset,
 * - the number of active dimensions (`rank`).
 *
 * @invariant `0 <= rank && rank <= MAX_DIMS`
 * @invariant Elements in `shape` and `strides` are only meaningful for the first `rank` active
 * entries
 */
struct TensorLayout {
	/**
	 * @brief Shape of the tensor.
	 *
	 * The shape values describe the extent of each dimension.
	 *
	 * @note Only the entries corresponding to the active rank are meaningful.
	 */
	std::array<size_t, MAX_DIMS> shape;

	/**
	 * @brief Strides for each tensor dimension.
	 *
	 * A stride describes how many elements must be skipped to move to the next index
	 * along a given dimension.
	 *
	 * @note Only the entries corresponding to the active rank are meaningful.
	 */
	std::array<size_t, MAX_DIMS> strides;

	/**
	 * @brief Number of elements skipped before indexing starts.
	 */
	size_t offset = 0;

	/**
	 * @brief Number of active dimensions in the layout.
	 *
	 * Only the first `rank` entries of `shape` and `strides` are considered valid.
	 */
	size_t rank = 0;

	/**
	 * @brief Constructs an empty tensor layout.
	 *
	 * @post `rank == 0`
	 * @post `offset == 0`
	 */
	TensorLayout() : shape{}, strides{} {}

	/**
	 * @brief Constructs a contiguous tensor layout from a shape.
	 *
	 * The layout receives contiguous strides inferred from `_shape`.
	 *
	 * @pre `_shape.getNumDims() <= MAX_DIMS`
	 * @post `rank == _shape.getNumDims()`
	 * @post `offset == 0`
	 * @post `strides` contains contiguous strides for `_shape`
	 *
	 * @param _shape Shape of the tensor layout.
	 */
	TensorLayout(const Tensor::Shape& _shape);

	/**
	 * @brief Constructs a tensor layout from a shape and explicit strides.
	 *
	 * @pre `_shape.getNumDims() <= MAX_DIMS`
	 * @pre `_strides.size() == _shape.getNumDims()`
	 * @post `rank == _shape.getNumDims()`
	 * @post `offset == 0`
	 *
	 * @param _shape Shape of the tensor layout.
	 * @param _strides Explicit strides for each dimension.
	 */
	TensorLayout(const Tensor::Shape& _shape, const std::vector<size_t>& _strides);

	/**
	 * @brief Constructs a tensor layout from a shape, explicit strides, and offset.
	 *
	 * @pre `_shape.getNumDims() <= MAX_DIMS`
	 * @pre `_strides.size() == _shape.getNumDims()`
	 * @post `rank == _shape.getNumDims()`
	 * @post `offset == _offset`
	 *
	 * @param _shape Shape of the tensor layout.
	 * @param _strides Explicit strides for each dimension.
	 * @param _offset Storage offset before indexing begins.
	 */
	TensorLayout(const Tensor::Shape& _shape, const std::vector<size_t>& _strides, size_t _offset);

	/**
	 * @brief Constructs a tensor layout from raw storage.
	 *
	 * @pre `_rank <= MAX_DIMS`
	 * @post `offset == _offset`
	 *
	 * @param shape Shape values.
	 * @param strides Stride values.
	 * @param offset Storage offset.
	 * @param rank Number of active dimensions.
	 */
	TensorLayout(std::array<size_t, MAX_DIMS> _shape, std::array<size_t, MAX_DIMS> _strides,
	             size_t _offset, size_t _rank);

	/**
	 * @brief Checks whether two tensor layouts are equal.
	 *
	 * Two layouts are equal if they have the same:
	 * - rank
	 * - offset
	 * - active shape values
	 * - active stride values
	 *
	 * Only the active `rank` entries are compared.
	 *
	 * @param rhs The layout to compare against.
	 * @return `true` if the layouts are equal, otherwise `false`.
	 */
	bool operator==(const TensorLayout& rhs) const;

	/**
	 * @brief Checks whether two tensor layouts are not equal.
	 *
	 * Equivalent to `!(*this == rhs)`.
	 *
	 * @param rhs The layout to compare against.
	 * @return `true` if the layouts are not equal, otherwise `false`.
	 */
	bool operator!=(const TensorLayout& rhs) const;
};

/**
 * @brief Converts a linear index to a physical offset in memory based on the given tensor layout.
 *
 * Accounts for the shape, strides, and offset defined in the layout to compute the correct element
 * offset.
 *
 * @param linear Linear index. This is the index as if the tensor were stored contiguously in
 * memory.
 * @param L Tensor layout describing the arrangement of the tensor.
 * @return Element offset corresponding to the linear index in the layout.
 */
static inline size_t physicalOffset(size_t linear, const TensorLayout& L) {
	size_t off = L.offset;
	for (int d = L.rank - 1; d >= 0; d--) {
		size_t coord = linear % L.shape[d];
		linear /= L.shape[d];
		off += coord * L.strides[d];
	}
	return off;
}

#endif  // TENSOR_LAYOUT_H