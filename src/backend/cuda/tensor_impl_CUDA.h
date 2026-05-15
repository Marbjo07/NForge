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


    void iadd(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, const TensorLayout& rhsLayout) override;    
    void isub(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, const TensorLayout& rhsLayout) override;    
    void imul(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, const TensorLayout& rhsLayout) override;
    void idiv(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, const TensorLayout& rhsLayout) override;

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
    float* d_data;

    const Tensor::CUDAImpl* cast(const Tensor::Impl* p) const;

    
    template <typename Kernel>
    std::unique_ptr<Tensor::Impl> applyKernel(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                                              const TensorLayout& rhsLayout, const TensorLayout& outLayout, Kernel kernel) const;
      
    template <typename Kernel>
    void applyInplaceKernel(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                            const TensorLayout& rhsLayout, Kernel kernel);

    // reduction must be associative
    template <typename Kernel>
    std::unique_ptr<Tensor::Impl> applyReductionKernel(const TensorLayout& layout, const TensorLayout& blockLayout,
                                                       const TensorLayout& outLayout, float initValue, Kernel kernel) const;

};

#endif  // TENSOR_IMPL_CUDA_H