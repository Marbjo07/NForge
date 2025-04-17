#include "Tensor.h"
#include "TensorBackend.h"
#include "ErrorHandler.h"

Tensor::Tensor(std::vector<size_t> shape, Backend backend)
	: m_shape(shape), m_backend(backend) {

	// calculate the number of elements
	m_size = this->getNumberOfElements();
	size_t mallocSize = m_size * sizeof(float);
	
	if (backend == Backend::CUDA) {
		cudaMalloc(&m_data, mallocSize);
		
		ErrorHandler::cudaSyncLog("unable to allocate tensor of size: " + std::to_string(mallocSize / 1024 / 1024) + "mb");
	}
	else {
		m_data = (float*)malloc(m_size * sizeof(float));
	}
	
	this->fillAll(0);
}


Tensor::Tensor(std::vector<size_t> shape, float value, Backend backend) 
	: Tensor(shape, backend) { 
	this->fillAll(value);
}

Tensor::Tensor(Tensor&& other) noexcept
	: m_shape(std::move(other.m_shape)), m_backend(other.m_backend), m_size(other.m_size), m_data(other.m_data) {
	// remove point to stop double free
	other.m_data = nullptr;
}


Tensor::Tensor(const Tensor& other)
	: m_shape(other.m_shape), m_backend(other.m_backend), m_size(other.m_size) {
	// allocate new memory
	m_data = (float*)malloc(m_size * sizeof(float));

	if (m_data == nullptr) {
		LOG(FATAL) << "Memory allocation failed in copy constructor!";
		return;
	}

	// copy the data
	std::memcpy(m_data, other.m_data, m_size * sizeof(float));
}


Tensor::~Tensor() {
	if (m_backend == Backend::CUDA) {
		cudaFree(m_data); // TODO: use cudaFreeAsync(...) ?
	}
	else {
		free(m_data);
	}
}


// Move data between CPU and CUDA
void Tensor::to(Backend newBackend) {
	if (m_backend == newBackend) return;

	if (newBackend == Backend::CUDA) {
		// copy from CPU to CUDA
		float* cudaData;
		cudaMalloc(&cudaData, m_size * sizeof(float));
		cudaMemcpy(cudaData, m_data, m_size * sizeof(float), cudaMemcpyHostToDevice);
		cudaFree(m_data);
		m_data = cudaData;
		
		ErrorHandler::cudaSyncLog("Unable to move tensor from CPU to CUDA");
	}
	else {
		// copy from CUDA to CPU
		float* cpuData = (float*)malloc(m_size * sizeof(float));
		cudaMemcpy(cpuData, m_data, m_size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(m_data);
		m_data = cpuData;
		
		ErrorHandler::cudaSyncLog("Unable to move tensor from CUDA to CPU");
	}
}

size_t Tensor::getNumberOfElements() const {
	size_t numberOfElements = 1;
	for (size_t axis : m_shape) {
		numberOfElements *= axis;
	}
	return numberOfElements;
}


void Tensor::fillAll(const float value) {
	if (m_backend == Backend::CUDA) {
		size_t mallocSize = m_size * sizeof(float);
		cudaMemset(&m_data, 0, mallocSize);

		ErrorHandler::cudaSyncLog("tried filling tensor of shape " + getShapeAsString() + " with value " + std::to_string(value));
	}
	else {
		std::fill((float*)m_data, (float*)m_data + m_size, value);
	}
}

void Tensor::print() const {
	LOG(INFO) << "Tensor content: " << getDataAsString();
}

std::string Tensor::getShapeAsString() const {
	std::string shapeString = "[";
	for (size_t i = 0; i < m_shape.size(); ++i) {
		shapeString += std::to_string(m_shape[i]);
		if (i != m_shape.size() - 1) {
			shapeString += ", ";
		}
	}
	shapeString += "]";
	return shapeString;
}

std::string Tensor::getBackendAsString() const {
	switch (m_backend) {
	case Backend::CPU:
		return "CPU";
	case Backend::CUDA:
		return "CUDA";
	default:
		return "Unknown";
	}
}

std::string Tensor::getDataAsString() const {
	std::string tensorContent = "{ ";

	if (m_backend == Backend::CUDA) {
		// wait for operations to finish
		cudaDeviceSynchronize();
		ErrorHandler::cudaSyncLog("Caught error when syncing before print!");

		// copy data from gpu 
		float* cpuData = (float*)malloc(m_size * sizeof(float));
		if (cpuData == nullptr) {
			LOG(FATAL) << "Memory allocation failed when printing tensor!";
			return "{}";
		}
		cudaMemcpy(cpuData, m_data, m_size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		ErrorHandler::cudaSyncLog("Unable to copy tensor when printing.");

		// create string of content
		for (size_t i = 0; i < m_size; ++i) {
			tensorContent += std::to_string(cpuData[i]);
			if (i != m_size - 1) {
				tensorContent += ", ";
			}
		}

		// free temporary tensor
		free(cpuData);
	}
	else {
		// create string of content
		for (size_t i = 0; i < m_size; ++i) {
			tensorContent += std::to_string(m_data[i]);
			if (i != m_size - 1) {
				tensorContent += ", ";
			}
		}
	}

	// finish string and print
	tensorContent += "}";
	return tensorContent;
}

// =====================
// Overloaded operators
// =====================

Tensor Tensor::operator+(const Tensor& other) const {
	if (m_backend == Backend::CUDA) {
		return backend::cuda::pointwise::addition(*this, other);
	}
	return backend::cpu::pointwise::addition(*this, other);
}

Tensor Tensor::operator-(const Tensor& other) const {
	if (m_backend == Backend::CUDA) {
		return backend::cuda::pointwise::subtraction(*this, other);
	}
	return backend::cpu::pointwise::subtraction(*this, other);
}

Tensor Tensor::operator*(const Tensor& other) const {
	if (m_backend == Backend::CUDA) {
		return backend::cuda::pointwise::multiplication(*this, other);
	}
	return backend::cpu::pointwise::multiplication(*this, other);
}

Tensor Tensor::operator/(const Tensor& other) const {
	if (m_backend == Backend::CUDA) {
		return backend::cuda::pointwise::division(*this, other);
	}
	return backend::cpu::pointwise::division(*this, other);
}