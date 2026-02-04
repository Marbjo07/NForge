#ifndef TENSOR_IMPL_CPU_H
#define TENSOR_IMPL_CPU_H

#include "nforge/tensor.h"
#include "nforge/tensor_shape.h"

class Tensor::CPUImpl : public Tensor::Impl {
public:
	CPUImpl(const Tensor::Shape &shape);
	CPUImpl(const Tensor::Shape &shape, float value);
	~CPUImpl();

	void fillAll(float value) override;
	void fillRand() override;

	void print() const override;
	void print(const std::vector<size_t> &position) const override;

    size_t numElements() const override;
    Tensor::Shape shape() const override;

    const float* data() const override;
    std::vector<float> toVector() const override;
	std::string toString() const override;

	std::unique_ptr<Tensor::Impl> clone() const override;

	std::unique_ptr<Tensor::Impl> get(size_t idx) const override;
	void set(const std::vector<size_t> &position, const Tensor::Impl &other) override;

	bool compare(const std::vector<size_t> &position, const Tensor::Impl &other) const override;
	bool compare(const std::vector<size_t> &position, const Tensor::Impl &other, const std::vector<size_t> &otherPosition) const override;

	std::unique_ptr<Tensor::Impl> add(const Tensor::Impl &other) const override;
	std::unique_ptr<Tensor::Impl> sub(const Tensor::Impl &other) const override;
	std::unique_ptr<Tensor::Impl> mul(const Tensor::Impl &other) const override;
	std::unique_ptr<Tensor::Impl> div(const Tensor::Impl &other) const override;
	std::unique_ptr<Tensor::Impl> pow(unsigned int exponent) const override;

	bool operator==(const Tensor::Impl &other) const override;
	bool operator!=(const Tensor::Impl &other) const override;

private:
	Tensor::Shape m_shape;
	std::vector<float> m_data;
};

#endif // TENSOR_IMPL_CPU_H