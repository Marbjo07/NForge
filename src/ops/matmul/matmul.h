#ifndef NFORGE_OPS_MATMUL_H
#define NFORGE_OPS_MATMUL_H

#include "nforge/core/tensor.h"
#include "nforge/core/tensor_view.h"

namespace nforge {

Tensor matmul(const Tensor::View& lhs, const Tensor::View& rhs);

};  // namespace nforge

#endif  // NFORGE_OPS_MATMUL_H