#ifndef TENSOR_IMPL_CUDA_H
#define TENSOR_IMPL_CUDA_H

#include "../tensor_impl.h"
#include "nforge/core/tensor_shape.h"

class Tensor::CUDAImpl : public Tensor::Impl {
   public:
    CUDAImpl(const Tensor::Shape& shape);
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
    void set(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
             const TensorLayout& rhsLayout) override;

    // Block comparisons
    bool compare(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                 const TensorLayout& rhsLayout) const override;

    // Element wise binary tensor operations
    std::unique_ptr<Tensor::Impl> add(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                                      const TensorLayout& rhsLayout, const TensorLayout& outLayout) const override;
    
    std::unique_ptr<Tensor::Impl> sub(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                                      const TensorLayout& rhsLayout, const TensorLayout& outLayout) const override;
    
    std::unique_ptr<Tensor::Impl> mul(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                                      const TensorLayout& rhsLayout, const TensorLayout& outLayout) const override;

    std::unique_ptr<Tensor::Impl> div(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                                      const TensorLayout& rhsLayout, const TensorLayout& outLayout) const override;
   private:
    Tensor::Shape m_shape;
    float* d_data;

    const Tensor::CUDAImpl* cast(const Tensor::Impl* p) const;

    // res[i] = kernel(lhs[i + lhsOffset], rhs[i + rhsOffset]), for all i, (0 <= i < count)
    template<typename Kernel>
    std::unique_ptr<Tensor::Impl> applyKernel(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count, Kernel kernel) const;
};

#endif  // TENSOR_IMPL_CPU_H