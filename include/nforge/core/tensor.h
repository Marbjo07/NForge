#ifndef TENSOR_H
#define TENSOR_H

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

enum class Backend {
    CPU,
    CUDA
};

class Tensor {
   public:
    class Impl;
    class CPUImpl;
    class CUDAImpl;

    class View;
    class Shape;

   public:
    Tensor(const Tensor::Shape& shape, Backend backend = Backend::CPU);
    Tensor(const Tensor::Shape& shape, float value, Backend backend = Backend::CPU);
    Tensor(float value, Backend backend = Backend::CPU);
    Tensor(const Tensor& tensor);
    Tensor(std::unique_ptr<Tensor::Impl> impl, Backend backend = Backend::CPU);
    ~Tensor();

    // Switch between backends
    void to(Backend newBackend);

    // Fill all elements with a value
    void fillAll(float value);

    // Fill all elements with uniform real values in [-1, 1]
    void fillRand();

    // Print the tensor data
    void print() const;
    void print(const std::vector<size_t>& position) const;

    // Returns the shape of the tensor
    Tensor::Shape getShape() const;

    // Returns the data of the tensor as a string
    std::string getBackendString() const;

    // Returns backend enum
    Backend getBackend() const;

    // Returns the data of the tensor as a string
    std::string getDataString() const;

    // Returns the number of elements in the tensor
    size_t getNumElements() const;

    // Returns tensor data as a vector
    std::vector<float> toVector() const;

    // Set the specified block to another tensor
    void set(const std::vector<size_t>& position, const Tensor& other);
    void set(const std::vector<size_t>& position, const Tensor::View& other);

    bool compare(const Tensor& other) const;
    bool compare(const Tensor::View& other) const;

    // Compare the specified block to another tensor
    bool compare(const std::vector<size_t>& position, const Tensor& other) const;
    bool compare(const std::vector<size_t>& position, const Tensor::View& other) const;

    // Overloaded operators
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor::View operator[](size_t idx) const;

    Tensor operator=(const Tensor& other);

    bool operator==(const Tensor& other) const;
    bool operator==(const Tensor::View& other) const;
    bool operator!=(const Tensor& other) const;
    bool operator!=(const Tensor::View& other) const;

   private:
    Backend m_backend;
    std::unique_ptr<Impl> m_impl;

    // used in template for all the binary operations
    template <typename EqualOp, typename ScalarOp>
    Tensor applyBinaryOp(const Tensor& rhs, const std::string& opName, EqualOp equalOp, ScalarOp scalarOp) const;
};

#include "nforge/backend/tensor_impl.h"
#include "nforge/core/tensor_shape.h"
#include "nforge/core/tensor_view.h"

#endif  // TENSOR_H