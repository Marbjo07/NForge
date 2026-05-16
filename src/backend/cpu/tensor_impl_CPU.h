#ifndef TENSOR_IMPL_CPU_H
#define TENSOR_IMPL_CPU_H

#include "../tensor_impl.h"
#include "nforge/core/tensor_layout.h"
#include "nforge/core/tensor_shape.h"

class Tensor::CPUImpl : public Tensor::Impl {
public:
	CPUImpl(const Tensor::Shape& shape);
	CPUImpl(const Tensor::Shape& shape, float value);
	~CPUImpl();

	// Fill functions
	void fillAll(float value) override;
	void fillRand() override;

	// Printing
	void print() const override;
	void print(const std::vector<size_t>& position) const override;

	// Tensor shape
	size_t getNumElements() const override;
	Tensor::Shape getShape() const override;

	// Data transforms
	float* dataPtr() const;
	std::vector<float> toVector() const override;
	std::string toString() const override;

	std::unique_ptr<Tensor::Impl> clone() const override;

	// Assignments and indexing
	void set(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	         const TensorLayout& rhsLayout) override;

	// Block comparisons
	bool compare(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	             const TensorLayout& rhsLayout) const override;

	// Element wise binary tensor operations
	std::unique_ptr<Tensor::Impl> add(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	                                  const TensorLayout& rhsLayout,
	                                  const TensorLayout& outLayout) const override;

	std::unique_ptr<Tensor::Impl> sub(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	                                  const TensorLayout& rhsLayout,
	                                  const TensorLayout& outLayout) const override;

	std::unique_ptr<Tensor::Impl> mul(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	                                  const TensorLayout& rhsLayout,
	                                  const TensorLayout& outLayout) const override;

	std::unique_ptr<Tensor::Impl> div(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	                                  const TensorLayout& rhsLayout,
	                                  const TensorLayout& outLayout) const override;

	void iadd(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	          const TensorLayout& rhsLayout) override;

	void isub(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	          const TensorLayout& rhsLayout) override;

	void imul(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	          const TensorLayout& rhsLayout) override;

	void idiv(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	          const TensorLayout& rhsLayout) override;

	std::unique_ptr<Tensor::Impl> sum(const TensorLayout& layout, const TensorLayout& blockLayout,
	                                  const TensorLayout& outLayout) const override;

	std::unique_ptr<Tensor::Impl> min(const TensorLayout& layout, const TensorLayout& blockLayout,
	                                  const TensorLayout& outLayout) const override;

	std::unique_ptr<Tensor::Impl> max(const TensorLayout& layout, const TensorLayout& blockLayout,
	                                  const TensorLayout& outLayout) const override;

	std::unique_ptr<Tensor::Impl> prod(const TensorLayout& layout, const TensorLayout& blockLayout,
	                                   const TensorLayout& outLayout) const override;

	std::unique_ptr<Tensor::Impl> norm(const TensorLayout& layout) const override;

private:
	Tensor::Shape m_shape;
	std::vector<float> m_data;

	template <typename BinaryOp>
	std::unique_ptr<Tensor::Impl> applyBinaryOp(const TensorLayout& lhsLayout,
	                                            const Tensor::Impl* rhsImpl,
	                                            const TensorLayout& rhsLayout,
	                                            const TensorLayout& outLayout, BinaryOp op) const;

	template <typename BinaryOp>
	void applyInplaceBinaryOp(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	                          const TensorLayout& rhsLayout, BinaryOp op);

	// reduction must be associative
	template <typename ReductionOp>
	std::unique_ptr<Tensor::Impl> applyReductionOp(const TensorLayout& layout,
	                                               const TensorLayout& blockLayout,
	                                               const TensorLayout& outLayout,
	                                               ReductionOp op) const;
};

#endif  // TENSOR_IMPL_CPU_H