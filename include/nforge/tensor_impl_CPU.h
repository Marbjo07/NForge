#ifndef TENSOR_IMPL_CPU_H
#define TENSOR_IMPL_CPU_H

#include "nforge/tensor.h"

class Tensor::CPUImpl : public Tensor::Impl {
public:
	CPUImpl(const std::vector<size_t>& shape);
	CPUImpl(const std::vector<size_t>& shape, float value);
	~CPUImpl();

	void fillAll(float value) override;
	void fillRand() override;

	void print() const override;
	void print(const std::vector<size_t>& id) const override;

	std::string getShapeAsString() const override;
	std::string getDataAsString() const override;

	size_t getNumberOfElements() const override;

	std::vector<float> getAsVector() const override;
	std::vector<size_t> getShape() const override;

	const float* getDataPointer() const override;

	std::unique_ptr<Tensor::Impl> clone() const override;

	std::unique_ptr<Tensor::Impl> get(size_t idx) const override;
	void set(const std::vector<size_t>& position, const Tensor::Impl& other) override;

	std::unique_ptr<Tensor::Impl> add(const Tensor::Impl& other) const override;
	std::unique_ptr<Tensor::Impl> sub(const Tensor::Impl& other) const override;
	std::unique_ptr<Tensor::Impl> mul(const Tensor::Impl& other) const override;
	std::unique_ptr<Tensor::Impl> div(const Tensor::Impl& other) const override;

	bool operator==(const Tensor::Impl& other) const override;
	bool operator!=(const Tensor::Impl& other) const override;

private:
	std::vector<size_t> m_shape;
	std::vector<float> m_data;
};

#endif // TENSOR_IMPL_CPU_H