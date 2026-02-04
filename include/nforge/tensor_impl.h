#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include "nforge/tensor.h"

class Tensor::Impl {
public:
    Impl() = default;
    virtual ~Impl() = default;

	virtual void fillAll(float value) = 0;
    virtual void fillRand() = 0;

	virtual void print() const = 0;
    virtual void print(const std::vector<size_t>& position) const = 0;

    virtual size_t numElements() const = 0;
    virtual Tensor::Shape shape() const = 0;
    
    virtual const float* data() const = 0;
    virtual std::vector<float> toVector() const = 0;
	virtual std::string toString() const = 0;

    virtual std::unique_ptr<Tensor::Impl> clone() const = 0;

    virtual std::unique_ptr<Tensor::Impl> get(size_t idx) const = 0;
    virtual void set(const std::vector<size_t>& position, const Tensor::Impl& other) = 0;

    virtual bool compare(const std::vector<size_t>& position, const Tensor::Impl& other) const = 0;
    virtual bool compare(const std::vector<size_t>& position, const Tensor::Impl& other, const std::vector<size_t>& otherIdx) const = 0;

	virtual std::unique_ptr<Tensor::Impl> add(const Tensor::Impl& other) const = 0;
	virtual std::unique_ptr<Tensor::Impl> sub(const Tensor::Impl& other) const = 0;
	virtual std::unique_ptr<Tensor::Impl> mul(const Tensor::Impl& other) const = 0;
	virtual std::unique_ptr<Tensor::Impl> div(const Tensor::Impl& other) const = 0;
    virtual std::unique_ptr<Tensor::Impl> pow(unsigned int exponent) const = 0;

    virtual bool operator==(const Tensor::Impl& other) const = 0;
    virtual bool operator!=(const Tensor::Impl& other) const = 0;
};

#endif // TENSOR_IMPL_H