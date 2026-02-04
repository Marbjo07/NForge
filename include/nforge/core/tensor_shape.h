#ifndef TENSOR_SHAPE_H
#define TENSOR_SHAPE_H

#include <vector>
#include <string>

#include "nforge/core/tensor.h"

class Tensor::Shape {
public:
    Shape() = default;
    Shape(const std::vector<size_t>& dims);
    Shape(const std::initializer_list<size_t>& dims);

    bool operator==(const Tensor::Shape &other) const;
    bool operator!=(const Tensor::Shape &other) const;

    // Access and properties
    size_t dims() const;
    size_t operator[](size_t index) const;
    size_t totalSize() const;
    bool isScalar() const;

    // Shape modifications
    Tensor::Shape removeLeadingDimension() const;
    Tensor::Shape slice(size_t start, size_t end) const;

    // Utility methods
    std::string toString() const;
    std::vector<size_t> toVector() const;
    std::vector<size_t> withoutTrailingOnes() const;

private:
    std::vector<size_t> m_dimensions;
};

#endif // TENSOR_SHAPE_H