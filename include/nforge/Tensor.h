#pragma once
#include "pch.h"

enum class Backend { CPU, CUDA };

class Tensor {
private:
    Backend m_backend;

public:
    Tensor(std::vector<size_t> shape, Backend backend);
    Tensor(std::vector<size_t> shape, float value, Backend backend);
    Tensor(Tensor&& other) noexcept;
    Tensor(const Tensor& other);
    ~Tensor();
    
    // Move data between CPU and CUDA
    void to(Backend newBackend);
    
    // Returns the number of elements in the tensor
    size_t getNumberOfElements() const; 
    
    // Fill all elements with a value
    void fillAll(const float value); 
    
    // Print the tensor data
    void print() const; 

    // Returns the shape of the tensor as a string
	std::string getShapeAsString() const;

	// Returns the data of the tensor as a string
    std::string getBackendAsString() const;

	// Returns the data of the tensor as a string
	std::string getDataAsString() const;

    // Overloaded operators
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    std::vector<size_t> m_shape; // Size of each dimension
    size_t m_size; // Element count
    float* m_data;  // Pointer to allocated memory (CPU or CUDA)
};