#pragma once
#include "Tensor.h"

namespace backend {
	namespace cuda {
		namespace pointwise {
			Tensor addition(const Tensor& a, const Tensor& b);
			Tensor subtraction(const Tensor& a, const Tensor& b);
			Tensor multiplication(const Tensor& a, const Tensor& b);
			Tensor division(const Tensor& a, const Tensor& b);
		}
	}

	namespace cpu {
		namespace pointwise {
			Tensor addition(const Tensor& a, const Tensor& b);
			Tensor subtraction(const Tensor& a, const Tensor& b);
			Tensor multiplication(const Tensor& a, const Tensor& b);
			Tensor division(const Tensor& a, const Tensor& b);
		}
	}
}