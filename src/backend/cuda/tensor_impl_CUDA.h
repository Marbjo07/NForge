#ifndef TENSOR_IMPL_CUDA_H
#define TENSOR_IMPL_CUDA_H

#include "../tensor_impl.h"
#include "nforge/core/tensor_shape.h"

/// CUDA implementation of Tensor::Impl, backed by device memory.
///
/// All operations launch CUDA kernels using TensorLayout descriptors.
/// The caller is responsible for layout validity, see Tensor::Impl.
///
/// Overridden methods follow the same semantics documented in Tensor::Impl.
class Tensor::CUDAImpl : public Tensor::Impl {
public:
	CUDAImpl(const Tensor::Shape& shape);
	~CUDAImpl();

	void fillAll(float value) override;
	void fillRand() override;

	void print() const override;
	void print(const std::vector<size_t>& position) const override;

	size_t getNumElements() const override;
	Tensor::Shape getShape() const override;

	/// Returns a raw pointer to the device data buffer.
	float* dataPtr() const;
	std::vector<float> toVector() const override;
	std::string toString() const override;

	std::unique_ptr<Tensor::Impl> clone() const override;

	void copyFromHost(const float* data, size_t count) override;

	void set(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	         const TensorLayout& rhsLayout) override;

	bool compare(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	             const TensorLayout& rhsLayout) const override;

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

	std::unique_ptr<Tensor::Impl> all(const TensorLayout& layout, const TensorLayout& blockLayout,
	                                  const TensorLayout& outLayout) const override;

	std::unique_ptr<Tensor::Impl> any(const TensorLayout& layout, const TensorLayout& blockLayout,
	                                  const TensorLayout& outLayout) const override;

	std::unique_ptr<Tensor::Impl> matmul(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	                                     const TensorLayout& rhsLayout,
	                                     const TensorLayout& outLayout, size_t batch, size_t m,
	                                     size_t k, size_t p) const override;


	std::unique_ptr<Tensor::Impl> less(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	                                   const TensorLayout& rhsLayout,
	                                   const TensorLayout& outLayout) const override;

	std::unique_ptr<Tensor::Impl> lessEqual(const TensorLayout& lhsLayout,
	                                        const Tensor::Impl* rhsImpl,
	                                        const TensorLayout& rhsLayout,
	                                        const TensorLayout& outLayout) const override;

	std::unique_ptr<Tensor::Impl> greater(const TensorLayout& lhsLayout,
	                                      const Tensor::Impl* rhsImpl,
	                                      const TensorLayout& rhsLayout,
	                                      const TensorLayout& outLayout) const override;

	std::unique_ptr<Tensor::Impl> greaterEqual(const TensorLayout& lhsLayout,
	                                           const Tensor::Impl* rhsImpl,
	                                           const TensorLayout& rhsLayout,
	                                           const TensorLayout& outLayout) const override;

	std::unique_ptr<Tensor::Impl> isClose(const TensorLayout& lhsLayout,
	                                      const Tensor::Impl* rhsImpl,
	                                      const TensorLayout& rhsLayout,
	                                      const TensorLayout& outLayout,
	                                      float tolerance) const override;

private:
	Tensor::Shape m_shape;
	float* d_data;

	/// Downcasts a generic Impl pointer to CUDAImpl. Asserts the type matches.
	const Tensor::CUDAImpl* cast(const Tensor::Impl* p) const;

	template <typename Kernel>
	std::unique_ptr<Tensor::Impl> applyKernel(const TensorLayout& lhsLayout,
	                                          const Tensor::Impl* rhsImpl,
	                                          const TensorLayout& rhsLayout,
	                                          const TensorLayout& outLayout, Kernel kernel) const;

	template <typename Kernel>
	void applyInplaceKernel(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	                        const TensorLayout& rhsLayout, Kernel kernel);


	// kernel must be associative
	template <typename Kernel>
	std::unique_ptr<Tensor::Impl> applyReductionKernel(const TensorLayout& layout,
	                                                   const TensorLayout& blockLayout,
	                                                   const TensorLayout& outLayout,
	                                                   float initValue, Kernel kernel) const;
};

#endif  // TENSOR_IMPL_CUDA_H