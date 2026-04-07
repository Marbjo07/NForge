#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include "nforge/core/tensor.h"

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
    virtual void set(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) = 0;

    // Comparisons
    // Block comparisons
    virtual bool compare(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const = 0;

    // Element wise binary tensor operations
    virtual std::unique_ptr<Tensor::Impl> add(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                                              const TensorLayout& rhsLayout, const TensorLayout& outLayout) const = 0;
    
    virtual std::unique_ptr<Tensor::Impl> sub(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                                              const TensorLayout& rhsLayout, const TensorLayout& outLayout) const = 0;
    
    virtual std::unique_ptr<Tensor::Impl> mul(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                                              const TensorLayout& rhsLayout, const TensorLayout& outLayout) const = 0;

    virtual std::unique_ptr<Tensor::Impl> div(const TensorLayout& lhsLayout, const Tensor::Impl* rhsImpl, 
                                              const TensorLayout& rhsLayout, const TensorLayout& outLayout) const = 0;

};

#endif  // TENSOR_IMPL_H