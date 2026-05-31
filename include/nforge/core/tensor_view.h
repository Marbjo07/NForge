#ifndef TENSOR_VIEW_H
#define TENSOR_VIEW_H

#include "nforge/core/tensor.h"
#include "nforge/core/tensor_shape.h"

class Tensor::View {
public:
	View(Tensor& parent);
	View(Tensor& parent, const std::vector<size_t>& position);
	View(Tensor& parent, const std::vector<size_t>& position, const TensorLayout& layout);

	// implicit casting
	View(const Tensor& parent);

	static Tensor::View broadcast(Tensor& source, const Tensor::Shape& shape);
	static Tensor::View subsample(const View& src, const std::vector<size_t>& factors);

	void print() const;

	// referenced tensor
	inline Tensor& getParent() const { return m_parent; }

	// position of this view in the tensor it references
	inline std::vector<size_t> getPosition() const { return m_position; }

	// number of elements preceding this view
	inline size_t getOffset() const { return m_layout.offset; }

	// shape of the view
	inline Tensor::Shape getShape() const { return Tensor::Shape(m_layout); }

	// returns the stride for each dim, not the underlying stride
	std::vector<size_t> getStride() const;

	// returns the underlying physical layout
	const TensorLayout& getLayout() const { return m_layout; }

	// creates a copy of the viewed tensor
	Tensor copy() const;

	Tensor operator+(const Tensor::View& rhs) const;
	Tensor operator-(const Tensor::View& rhs) const;
	Tensor operator*(const Tensor::View& rhs) const;
	Tensor operator/(const Tensor::View& rhs) const;

	void operator+=(const Tensor::View& rhs);
	void operator-=(const Tensor::View& rhs);
	void operator*=(const Tensor::View& rhs);
	void operator/=(const Tensor::View& rhs);

	Tensor::View operator=(const Tensor& rhs);
	Tensor::View operator=(const Tensor::View& rhs);
	Tensor::View operator=(float scalar);

	Tensor::View operator[](size_t idx) const;

	Tensor matmul(const Tensor::View& rhs) const;

	Tensor::View subsample(std::vector<size_t> strides) const;

	bool operator==(const Tensor& rhs) const;
	bool operator==(const Tensor::View& rhs) const;

	bool operator!=(const Tensor& rhs) const;
	bool operator!=(const Tensor::View& rhs) const;

	Tensor operator<(const Tensor::View& rhs) const;
	Tensor operator<=(const Tensor::View& rhs) const;
	Tensor operator>(const Tensor::View& rhs) const;
	Tensor operator>=(const Tensor::View& rhs) const;

	Tensor isClose(const Tensor::View& rhs, float tolerance = 1e-5f) const;

private:
	// resolves ambiguous overload with initializer list
	struct BroadcastTag {};

	// trusted, used by broadcast.
	View(Tensor& parent, const std::vector<size_t>& stride, const Tensor::Shape& shape,
	     BroadcastTag);

	Tensor& m_parent;
	std::vector<size_t> m_position;
	TensorLayout m_layout;  // relative to parent tensor.
};

#endif  // TENSOR_VIEW_H