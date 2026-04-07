#ifndef TENSOR_IMPL_CPU_H
#define TENSOR_IMPL_CPU_H

#include "../tensor_impl.h"
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
    void set(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) override;

    // Comparisons
    bool compare(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const override;

    // Element wise binary tensor operations
    std::unique_ptr<Tensor::Impl> add(size_t lhsOffset, size_t lhsStride, size_t lhsCount, const Tensor::Impl* rhs, size_t rhsOffset, size_t rhsStride, size_t rhsCount, size_t count) const override;
    std::unique_ptr<Tensor::Impl> sub(size_t lhsOffset, size_t lhsStride, size_t lhsCount, const Tensor::Impl* rhs, size_t rhsOffset, size_t rhsStride, size_t rhsCount, size_t count) const override;
    std::unique_ptr<Tensor::Impl> mul(size_t lhsOffset, size_t lhsStride, size_t lhsCount, const Tensor::Impl* rhs, size_t rhsOffset, size_t rhsStride, size_t rhsCount, size_t count) const override;
    std::unique_ptr<Tensor::Impl> div(size_t lhsOffset, size_t lhsStride, size_t lhsCount, const Tensor::Impl* rhs, size_t rhsOffset, size_t rhsStride, size_t rhsCount, size_t count) const override;

   private:
    Tensor::Shape m_shape;
    std::vector<float> m_data;
    
    // res[i] = lhs[i] + scalar
    template <typename ScalarOp>
    std::unique_ptr<Tensor::Impl> applyBinaryScalarOp(size_t lhsOffset, const Tensor::Impl* rhs, size_t count, ScalarOp scalarOp) const;

    // res[i] = lhs[i * stride] + rhs[i * stride]
    template <typename BinaryOp>
    std::unique_ptr<Tensor::Impl> applyBinaryOp(size_t lhsOffset, size_t lhsStride, size_t lhsCount, const Tensor::Impl* rhs, size_t rhsOffset, size_t rhsStride, size_t rhsCount, size_t count, BinaryOp binaryOp) const;
};

#endif  // TENSOR_IMPL_CPU_H