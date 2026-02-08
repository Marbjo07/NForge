#ifndef TENSOR_IMPL_CPU_H
#define TENSOR_IMPL_CPU_H

#include "nforge/core/tensor.h"
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
    std::unique_ptr<Tensor::Impl> get(size_t idx) const override;
    void set(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) override;

    // Comparisons
    bool compare(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const override;

    // Element wise binary tensor operations
    std::unique_ptr<Tensor::Impl> add(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const override;
    std::unique_ptr<Tensor::Impl> sub(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const override;
    std::unique_ptr<Tensor::Impl> mul(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const override;
    std::unique_ptr<Tensor::Impl> div(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count) const override;

    // Element wise binary tensor-scalar operations
	// Requires  rhs to be a scalar
    std::unique_ptr<Tensor::Impl> addScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const override;
    std::unique_ptr<Tensor::Impl> subScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const override;
    std::unique_ptr<Tensor::Impl> mulScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const override;
    std::unique_ptr<Tensor::Impl> divScalar(size_t lhsOffset, const Tensor::Impl* rhs, size_t count) const override;

   private:
    Tensor::Shape m_shape;
    std::vector<float> m_data;
    
    // res[i] = lhs[i] + scalar
    template <typename ScalarOp>
    std::unique_ptr<Tensor::Impl> applyBinaryScalarOp(size_t lhsOffset, const Tensor::Impl* rhs, size_t count, ScalarOp scalarOp) const;

    // res[i] = lhs[i] + rhs[i]
    template <typename BinaryOp>
    std::unique_ptr<Tensor::Impl> applyBinaryOp(size_t lhsOffset, const Tensor::Impl* rhs, size_t rhsOffset, size_t count, BinaryOp binaryOp) const;
};

#endif  // TENSOR_IMPL_CPU_H