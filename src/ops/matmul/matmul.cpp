#include "ops/matmul/matmul.h"

Tensor nforge::matmul(const Tensor::View& lhs, const Tensor::View& rhs) { return lhs.matmul(rhs); }
