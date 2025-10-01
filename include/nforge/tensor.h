#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <cassert>

#define LOG(x) std::cerr

enum class Backend { CPU };

class Tensor {
public:
    class Impl;
    class CPUImpl;
    class CUDAImpl;
    
    class View;

public:
    Tensor(const std::vector<size_t>& shape, Backend backend = Backend::CPU);
    Tensor(const std::vector<size_t>& shape, float value, Backend backend = Backend::CPU);
    Tensor(float value, Backend backend = Backend::CPU);
    Tensor(const Tensor& tensor);
    Tensor(std::unique_ptr<Tensor::Impl> impl);
    ~Tensor();
    
    // Move data between CPU and CUDA
    void to(Backend newBackend);
    
    // Fill all elements with a value
    void fillAll(float value);
    
    // Fill all elements with uniform real values in [-1, 1]
    void fillRand();
    
    // Print the tensor data
    void print() const; 
    void print(const std::vector<size_t>& idx) const;

    // Returns the shape of the tensor as a string
	std::string getShapeAsString() const;

	// Returns the data of the tensor as a string
    std::string getBackendAsString() const;

	// Returns the data of the tensor as a string
	std::string getDataAsString() const;
    
    // Returns the number of elements in the tensor
    size_t getNumberOfElements() const;
    
    // Returns tensor data as a vector 
    std::vector<float> getAsVector() const;

    // Set the indexed block to another tensor
    void set(const std::vector<size_t>& idx, const Tensor& other);

    // Returns the tensor raised to a positive integer value
    Tensor pow(unsigned int exponent) const;

    // Overloaded operators
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor::View operator[](size_t idx) const;

    Tensor operator=(const Tensor& other);

    bool operator==(const Tensor& other) const;
    bool operator!=(const Tensor& other) const;

private:
    Backend m_backend;
    std::unique_ptr<Impl> m_impl;
};

class Tensor::Impl {
public:
    Impl() = default;
    virtual ~Impl() = default;

	virtual void fillAll(float value) = 0;
    virtual void fillRand() = 0;

	virtual void print() const = 0;
    virtual void print(const std::vector<size_t>& idx) const = 0;

	virtual std::string getShapeAsString() const = 0;
	virtual std::string getDataAsString() const = 0;
	
    virtual size_t getNumberOfElements() const = 0;

    virtual std::vector<float> getAsVector() const = 0;
    virtual std::vector<size_t> getShape() const = 0;

    virtual const float* getDataPointer() const = 0;

    virtual std::unique_ptr<Tensor::Impl> clone() const = 0;

    virtual std::unique_ptr<Tensor::Impl> get(size_t idx) const = 0;
    virtual void set(const std::vector<size_t>& position, const Tensor::Impl& other) = 0;

	virtual std::unique_ptr<Tensor::Impl> add(const Tensor::Impl& other) const = 0;
	virtual std::unique_ptr<Tensor::Impl> sub(const Tensor::Impl& other) const = 0;
	virtual std::unique_ptr<Tensor::Impl> mul(const Tensor::Impl& other) const = 0;
	virtual std::unique_ptr<Tensor::Impl> div(const Tensor::Impl& other) const = 0;
    virtual std::unique_ptr<Tensor::Impl> pow(unsigned int exponent) const = 0;

    virtual bool operator==(const Tensor::Impl& other) const = 0;
    virtual bool operator!=(const Tensor::Impl& other) const = 0;
};

#include "nforge/tensor_view.h"

#endif // TENSOR_H