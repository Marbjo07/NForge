#ifndef TENSOR_IMPL_CUDA_H
#define TENSOR_IMPL_CUDA_H

#include "nforge/core/tensor.h"
#include "nforge/core/tensor_shape.h"

class Tensor::CUDAImpl : public Tensor::Impl {
   public:
    CUDAImpl(const Tensor::Shape& shape);
    CUDAImpl(const Tensor::Shape& shape, float value);
    ~CUDAImpl();

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
    void set(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) override;

    // Comparisons
    bool compare(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const override;

    // Element wise binary tensor operations
    std::unique_ptr<Tensor::Impl> add(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const override;
    std::unique_ptr<Tensor::Impl> sub(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const override;
    std::unique_ptr<Tensor::Impl> mul(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const override;
    std::unique_ptr<Tensor::Impl> div(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const override;

    // Element wise binary tensor-scalar operations
	// Requires rhs to be a scalar
    std::unique_ptr<Tensor::Impl> addScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const override;
    std::unique_ptr<Tensor::Impl> subScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const override;
    std::unique_ptr<Tensor::Impl> mulScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const override;
    std::unique_ptr<Tensor::Impl> divScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const override;

   private:
    Tensor::Shape m_shape;
    float* m_data;
};

#endif  // TENSOR_IMPL_CPU_H