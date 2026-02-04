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
    virtual size_t numElements() const = 0;
    virtual Tensor::Shape shape() const = 0;


    // Data transforms
    virtual const float* data() const = 0;
    virtual std::vector<float> toVector() const = 0;
	virtual std::string toString() const = 0;

    virtual std::unique_ptr<Tensor::Impl> clone() const = 0;


    // Assignments and indexing
    virtual std::unique_ptr<Tensor::Impl> get(size_t idx) const = 0;
    virtual void set(const std::vector<size_t>& position, const Tensor::Impl& other) = 0;


    // Comparisons
    // Block comparisons
    virtual bool compare(const std::vector<size_t>& position, const Tensor::Impl& other) const = 0;
    virtual bool compare(const std::vector<size_t>& position, const Tensor::Impl& other, const std::vector<size_t>& otherIdx) const = 0;

    // Tensor comparisons
    virtual bool operator==(const Tensor::Impl& other) const = 0;
    virtual bool operator!=(const Tensor::Impl& other) const = 0;


	// Element wise binary tensor operations
	virtual std::unique_ptr<Tensor::Impl> add(const Tensor::Impl& other) const = 0;
	virtual std::unique_ptr<Tensor::Impl> sub(const Tensor::Impl& other) const = 0;
	virtual std::unique_ptr<Tensor::Impl> mul(const Tensor::Impl& other) const = 0;
	virtual std::unique_ptr<Tensor::Impl> div(const Tensor::Impl& other) const = 0;


	// Element wise binary tensor-scalar operations
    // Requires the passed Tensor::Impl to be a scalar
	virtual std::unique_ptr<Tensor::Impl> addScalar(const Tensor::Impl& other) const = 0;
	virtual std::unique_ptr<Tensor::Impl> subScalar(const Tensor::Impl& other) const = 0;
	virtual std::unique_ptr<Tensor::Impl> mulScalar(const Tensor::Impl& other) const = 0;
	virtual std::unique_ptr<Tensor::Impl> divScalar(const Tensor::Impl& other) const = 0;
};

#endif // TENSOR_IMPL_H