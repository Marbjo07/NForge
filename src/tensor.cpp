#include "nforge/tensor.h"
#include "nforge/tensor_view.h"
#include "nforge/tensor_impl_CPU.h"

Tensor::Tensor(const std::vector<size_t>& shape, Backend backend) 
	: m_backend(backend) {
	if (backend == Backend::CPU) {
		m_impl = std::make_unique<Tensor::CPUImpl>(shape);
	}
	else {
		assert(false);
	}
}

Tensor::Tensor(const std::vector<size_t>& shape, float value, Backend backend) 
	: Tensor(shape, backend) {
	m_impl->fillAll(value);
}

Tensor::Tensor(float value, Backend backend) 
	: Tensor({1}, value, backend) {
}

Tensor::Tensor(const Tensor& other) 
	: m_backend(other.m_backend), m_impl(other.m_impl->clone()) {
}

Tensor::Tensor(std::unique_ptr<Tensor::Impl> impl)
	: m_impl(std::move(impl)) {
}

Tensor::~Tensor() {
}

void Tensor::fillAll(float value) {
	m_impl->fillAll(value);
}

void Tensor::fillRand() {
	m_impl->fillRand();
}

void Tensor::print() const {
	m_impl->print();
}

void Tensor::print(const std::vector<size_t>& idx) const {
	m_impl->print(idx);
}

std::string Tensor::getShapeAsString() const {
	return m_impl->getShapeAsString();
}

std::string Tensor::getBackendAsString() const {
	switch (m_backend) {
		case Backend::CPU:
			return "CPU";
		default:
			return "UNKNOWN";
	}
}

std::string Tensor::getDataAsString() const {
	return m_impl->getDataAsString();
}

size_t Tensor::getNumberOfElements() const {
	return m_impl->getNumberOfElements();
}

std::vector<float> Tensor::getAsVector() const {
	return m_impl->getAsVector();
}

void Tensor::set(const std::vector<size_t>& idx, const Tensor& other) {
	m_impl->set(idx, *other.m_impl.get());
}

bool Tensor::compare(const std::vector<size_t>& idx, const Tensor& other) const {
	return m_impl->compare(idx, *other.m_impl.get());
}

bool Tensor::compare(const std::vector<size_t>& idx, const Tensor::View& other) const {
	return m_impl->compare(idx, *other.getParent().m_impl.get(), other.getPosition());
}

Tensor Tensor::pow(unsigned int exponent) const {
	return Tensor(m_impl->pow(exponent));
}

Tensor Tensor::operator+(const Tensor& other) const {
	return Tensor(m_impl->add(*other.m_impl));
}

Tensor Tensor::operator-(const Tensor& other) const {
	return Tensor(m_impl->sub(*other.m_impl));
}

Tensor Tensor::operator*(const Tensor& other) const {
	return Tensor(m_impl->mul(*other.m_impl));
}

Tensor Tensor::operator/(const Tensor& other) const {
	return Tensor(m_impl->div(*other.m_impl));
}

Tensor::View Tensor::operator[](size_t idx) const {
	Tensor::View results((Tensor&)*this, {idx});
	return results;
}

Tensor Tensor::operator=(const Tensor& other) {
	this->m_impl = other.m_impl->clone();
	this->m_backend = other.m_backend;
	
	return *this;
}

bool Tensor::operator==(const Tensor& other) const {
	return m_backend == other.m_backend && m_impl->operator==(*other.m_impl);
}

bool Tensor::operator!=(const Tensor& other) const {
	return !operator==(other);
}
