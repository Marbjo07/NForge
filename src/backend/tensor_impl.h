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
    virtual std::unique_ptr<Tensor::Impl> add(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const = 0;
    virtual std::unique_ptr<Tensor::Impl> sub(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const = 0;
    virtual std::unique_ptr<Tensor::Impl> mul(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const = 0;
    virtual std::unique_ptr<Tensor::Impl> div(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const = 0;

    // Element wise binary tensor-scalar operations
    // Requires rhs to be a scalar
    virtual std::unique_ptr<Tensor::Impl> addScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const = 0;
    virtual std::unique_ptr<Tensor::Impl> subScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const = 0;
    virtual std::unique_ptr<Tensor::Impl> mulScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const = 0;
    virtual std::unique_ptr<Tensor::Impl> divScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const = 0;
};

#endif  // TENSOR_IMPL_H