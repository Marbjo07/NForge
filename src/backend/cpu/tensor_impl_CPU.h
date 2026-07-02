#ifndef TENSOR_IMPL_CPU_H
#define TENSOR_IMPL_CPU_H

#include "../tensor_impl.h"
#include "nforge/core/tensor_layout.h"
#include "nforge/core/tensor_shape.h"

/// CPU implementation of Tensor::Impl, backed by std::vector<float>.
///
/// All operations iterate over the data using TensorLayout descriptors.
/// The caller is responsible for layout validity, see Tensor::Impl.
///
/// Overridden methods follow the same semantics documented in Tensor::Impl.
class Tensor::CPUImpl : public Tensor::Impl {
public:
	CPUImpl(const Tensor::Shape& shape);
	CPUImpl(const Tensor::Shape& shape, float value);
	~CPUImpl();

	void fillAll(float value) override;
	void fillRand() override;

	void print() const override;
	void print(const std::vector<size_t>& position) const override;

	size_t getNumElements() const override;
	Tensor::Shape getShape() const override;

	/// Returns a raw pointer to the internal data buffer.
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


	std::unique_ptr<Tensor::Impl> equal(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	                                    const TensorLayout& rhsLayout,
	                                    const TensorLayout& outLayout) const override;

	std::unique_ptr<Tensor::Impl> notEqual(const TensorLayout& lhsLayout,
	                                       const Tensor::Impl* rhsImpl,
	                                       const TensorLayout& rhsLayout,
	                                       const TensorLayout& outLayout) const override;

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
	std::vector<float> m_data;

	template <typename BinaryOp>
	std::unique_ptr<Tensor::Impl> applyBinaryOp(const TensorLayout& lhsLayout,
	                                            const Tensor::Impl* rhsImpl,
	                                            const TensorLayout& rhsLayout,
	                                            const TensorLayout& outLayout, BinaryOp op) const;

	template <typename BinaryOp>
	void applyInplaceBinaryOp(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl,
	                          const TensorLayout& rhsLayout, BinaryOp op);


	struct Identity {
		template <typename T>
		constexpr T&& operator()(T&& t) const noexcept {
			return std::forward<T>(t);
		}
	};

	// reduction must be associative
	// x = f(x) must be true.
	// transform is applied to the first element, so transform(x) = f(x) must be true.
	template <typename ReductionOp, typename Transform = Identity>
	std::unique_ptr<Tensor::Impl> applyReductionOp(const TensorLayout& layout,
	                                               const TensorLayout& blockLayout,
	                                               const TensorLayout& outLayout, ReductionOp op,
	                                               Transform transform = {}) const;
};

#endif  // TENSOR_IMPL_CPU_H