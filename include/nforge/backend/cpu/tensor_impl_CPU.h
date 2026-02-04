#ifndef TENSOR_IMPL_CPU_H
#define TENSOR_IMPL_CPU_H

#include "nforge/core/tensor.h"
#include "nforge/core/tensor_shape.h"

class Tensor::CPUImpl : public Tensor::Impl {
public:
	CPUImpl(const Tensor::Shape &shape);
	CPUImpl(const Tensor::Shape &shape, float value);
	~CPUImpl();

	// Fill functions
	void fillAll(float value) override;
	void fillRand() override;


	// Printing
	void print() const override;
	void print(const std::vector<size_t> &position) const override;


	// Tensor shape
    size_t numElements() const override;
    Tensor::Shape shape() const override;


	// Data transforms
    const float* data() const override;
    std::vector<float> toVector() const override;
	std::string toString() const override;

	std::unique_ptr<Tensor::Impl> clone() const override;


    // Assignments and indexing
	std::unique_ptr<Tensor::Impl> get(size_t idx) const override;
	void set(const std::vector<size_t> &position, const Tensor::Impl &other) override;
	void set(const std::vector<size_t> &position, const Tensor::Impl &other, const std::vector<size_t> &otherPosition) override;


	// Comparisons
	bool compare(const std::vector<size_t> &position, const Tensor::Impl &other) const override;
	bool compare(const std::vector<size_t> &position, const Tensor::Impl &other, const std::vector<size_t> &otherPosition) const override;

	bool operator==(const Tensor::Impl &other) const override;
	bool operator!=(const Tensor::Impl &other) const override;


	// Element wise binary tensor operations
	std::unique_ptr<Tensor::Impl> add(const Tensor::Impl &other) const override;
	std::unique_ptr<Tensor::Impl> sub(const Tensor::Impl &other) const override;
	std::unique_ptr<Tensor::Impl> mul(const Tensor::Impl &other) const override;
	std::unique_ptr<Tensor::Impl> div(const Tensor::Impl &other) const override;


	// Element wise binary tensor-scalar operations
    // Requires the passed Tensor::Impl to be a scalar
	std::unique_ptr<Tensor::Impl> addScalar(const Tensor::Impl& scalar) const override;
	std::unique_ptr<Tensor::Impl> subScalar(const Tensor::Impl& scalar) const override;
	std::unique_ptr<Tensor::Impl> mulScalar(const Tensor::Impl& scalar) const override;
	std::unique_ptr<Tensor::Impl> divScalar(const Tensor::Impl& scalar) const override;

private:
	Tensor::Shape m_shape;
	std::vector<float> m_data;
};

#endif // TENSOR_IMPL_CPU_H