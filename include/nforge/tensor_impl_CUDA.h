#include "nforge/tensor.h"

#ifdef BUILD_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif // BUILD_WITH_CUDA

class Tensor::CUDAImpl : public Tensor::Impl {
public:
	CUDAImpl(const std::vector<size_t>& shape);
	CUDAImpl(const std::vector<size_t>& shape, float value);
	~CUDAImpl();

	void fillAll(const float value);
	void print() const;

	std::string getShapeAsString() const;

	std::string getDataAsString() const;

	Tensor::Impl operator+(const Tensor::Impl& other) const;
	Tensor::Impl operator-(const Tensor::Impl& other) const;
	Tensor::Impl operator*(const Tensor::Impl& other) const;
	Tensor::Impl operator/(const Tensor::Impl& other) const;

private:
	std::vector<size_t> m_shape;
};