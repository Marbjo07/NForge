#include "pch.h"

#include "Tensor.h"
#include "TensorBackend.h"
#include "ErrorHandler.h"

Tensor backend::cpu::pointwise::addition(const Tensor& a, const Tensor& b) {
	if (a.m_size != b.m_size) {
		LOG(ERROR) << "Mismatch between tensors when adding!";
		return a;
	}

	Tensor result(a.m_shape, 0.0f, Backend::CPU);

	for (int i = 0; i < a.m_size; i++) {
		result.m_data[i] = a.m_data[i] + b.m_data[i];
	}

	return result;
}

Tensor backend::cpu::pointwise::subtraction(const Tensor& a, const Tensor& b) {
	if (a.m_size != b.m_size) {
		LOG(ERROR) << "Mismatch between tensors when subtracting!";
		return a;
	}

	Tensor result(a.m_shape, 0.0f, Backend::CPU);

	for (int i = 0; i < a.m_size; i++) {
		result.m_data[i] = a.m_data[i] - b.m_data[i];
	}

	return result;
}

Tensor backend::cpu::pointwise::multiplication(const Tensor& a, const Tensor& b) {
	if (a.m_size != b.m_size) {
		LOG(ERROR) << "Mismatch between tensors when multiplying!";
		return a;
	}

	Tensor result(a.m_shape, 0.0f, Backend::CPU);

	for (int i = 0; i < a.m_size; i++) {
		result.m_data[i] = a.m_data[i] * b.m_data[i];
	}

	return result;
}

Tensor backend::cpu::pointwise::division(const Tensor& a, const Tensor& b) {
	if (a.m_size != b.m_size) {
		LOG(ERROR) << "Mismatch between tensors when dividing!";
		return a;
	}
	
	Tensor result(a.m_shape, 0.0f, Backend::CPU);

	for (int i = 0; i < a.m_size; i++) {
		result.m_data[i] = a.m_data[i] / b.m_data[i];
	}

	return result;
}