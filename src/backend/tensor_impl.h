#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include "nforge/core/tensor.h"
#include "nforge/core/tensor_layout.h"

class Tensor::Impl {
   public:
    Impl() = default;
    virtual ~Impl() = default;

    // Fill functions
    virtual void fillAll(float value) = 0;
    virtual void fillRand() = 0;

    // Printing
    virtual void print() const = 0;
    virtual void print(const std::vector<size_t>& position) const = 0;

    // Tensor shape
    virtual size_t getNumElements() const = 0;
    virtual Tensor::Shape getShape() const = 0;

    // Data transforms
    virtual std::vector<float> toVector() const = 0;
    virtual std::string toString() const = 0;

    virtual std::unique_ptr<Tensor::Impl> clone() const = 0;

    // Assignments and indexing
    virtual void set(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                     const TensorLayout& rhsLayout) = 0;

    // Block comparisons
    virtual bool compare(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                         const TensorLayout& rhsLayout) const = 0;

    // Element wise binary tensor operations
    virtual std::unique_ptr<Tensor::Impl> add(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                                              const TensorLayout& rhsLayout, const TensorLayout& outLayout) const = 0;
    
    virtual std::unique_ptr<Tensor::Impl> sub(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                                              const TensorLayout& rhsLayout, const TensorLayout& outLayout) const = 0;
    
    virtual std::unique_ptr<Tensor::Impl> mul(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                                              const TensorLayout& rhsLayout, const TensorLayout& outLayout) const = 0;

    virtual std::unique_ptr<Tensor::Impl> div(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                                              const TensorLayout& rhsLayout, const TensorLayout& outLayout) const = 0;


    virtual std::unique_ptr<Tensor::Impl> sum(const TensorLayout& layout, const TensorLayout& blockLayout, 
                                               const TensorLayout& outLayout) const = 0;

    virtual std::unique_ptr<Tensor::Impl> min(const TensorLayout& layout, const TensorLayout& blockLayout, 
                                               const TensorLayout& outLayout) const = 0;

    virtual std::unique_ptr<Tensor::Impl> max(const TensorLayout& layout, const TensorLayout& blockLayout, 
                                               const TensorLayout& outLayout) const = 0;

    virtual std::unique_ptr<Tensor::Impl> prod(const TensorLayout& layout, const TensorLayout& blockLayout, 
                                               const TensorLayout& outLayout) const = 0;
};

#endif  // TENSOR_IMPL_H