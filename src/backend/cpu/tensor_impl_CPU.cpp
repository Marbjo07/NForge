#include "nforge/core/tensor.h"
#include "nforge/backend/cpu/tensor_impl_CPU.h"

#include <algorithm>
#include <random>

Tensor::CPUImpl::CPUImpl(const Tensor::Shape& shape) 
	: m_shape(shape) {
	m_data.assign(m_shape.totalSize(), 0.0f);
}

Tensor::CPUImpl::CPUImpl(const Tensor::Shape& shape, float value) 
	: m_shape(shape) {
	m_data.assign(m_shape.totalSize(), value);
}

Tensor::CPUImpl::~CPUImpl() {
	m_data.clear();
	m_data.shrink_to_fit();
}

void Tensor::CPUImpl::fillAll(float value) {
	m_data.assign(m_data.size(), value);
}

void Tensor::CPUImpl::fillRand() {
	static std::mt19937 engine(std::random_device{}());
    static std::uniform_real_distribution<double> dist(-1.0, 1.0);

	auto gen = [&]() {
		return dist(engine);
	};

	std::generate(m_data.begin(), m_data.end(), gen);
}

void Tensor::CPUImpl::print() const {
	std::cout << "====================\n";
	std::cout << "Tensor[CPU], Data:\n";

	std::vector<size_t> numElementsInDimsCurAndBelow(m_shape.dims(), 1);
	for (int i = static_cast<int>(m_shape.dims()) - 1; i >= 0; i--) {
		numElementsInDimsCurAndBelow[i] *= m_shape[i];
		if (i != static_cast<int>(m_shape.dims()) - 1) {
			numElementsInDimsCurAndBelow[i] *= numElementsInDimsCurAndBelow[i + 1];
		}
	}

	for (size_t i = 0; i < m_data.size(); i++) {
		std::cout << m_data[i] << " ";
		for (size_t j = 0; j < m_shape.dims(); j++) {
			if (i % numElementsInDimsCurAndBelow[j] == numElementsInDimsCurAndBelow[j] - 1) {
				std::cout << "\n";
			}
		}
	}

	std::cout << "Shape: " << shape().toString() << "\n";
	std::cout << "====================\n";
}

void Tensor::CPUImpl::print(const std::vector<size_t>& position) const {
	std::cout << "====================\n";
	std::cout << "Tensor[CPU], Data:\n";

	std::vector<size_t> numElementsInDimsCurAndBelow(m_shape.dims(), 1);
	for (int i = static_cast<int>(m_shape.dims()) - 1; i >= 0; i--) {
		numElementsInDimsCurAndBelow[i] *= m_shape[i];
		if (i != static_cast<int>(m_shape.dims()) - 1) {
			numElementsInDimsCurAndBelow[i] *= numElementsInDimsCurAndBelow[i + 1];
		}
	}

	size_t blockSize = m_shape.slice(position.size(), m_shape.dims()).totalSize();

	size_t offsetCount = blockSize;
	for (size_t i = 0; i < position.size(); i++) {
		// if first slice in a dim
		if (offsetCount == 0) {
			offsetCount = blockSize;
		}

		offsetCount *= position[i];
	}

	for (size_t i = offsetCount; i < offsetCount + blockSize; i++) {
		std::cout << m_data[i] << " ";
		for (size_t j = 0; j < m_shape.dims(); j++) {
			if (i % numElementsInDimsCurAndBelow[j] == numElementsInDimsCurAndBelow[j] - 1) {
				std::cout << "\n";
			}
		}
	}
	std::cout << "\n";

	std::cout << "Shape: " << m_shape.slice(position.size(), m_shape.dims()).toString() << "\n";
	std::cout << "====================\n";
}

Tensor::Shape Tensor::CPUImpl::shape() const {
	return m_shape;
}

std::string Tensor::CPUImpl::toString() const {
	std::string out;

	out += "{ ";
	for (float element : m_data) {
		out += std::to_string(element) + " ";
	}
	out += "}";

	return out;
}

size_t Tensor::CPUImpl::numElements() const {
	size_t numImpliedByShape = m_shape.totalSize();
	size_t numInContainer = m_data.size();

	assert(numImpliedByShape == numInContainer);
	
	return numInContainer;
}

std::vector<float> Tensor::CPUImpl::toVector() const {
	return m_data;
}

const float* Tensor::CPUImpl::data() const {
	return m_data.data();
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::clone() const {
	return std::make_unique<CPUImpl>(*this);
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::get(size_t idx) const {
	// shape {3, 2, 5}
	// [[x, x, x, x, x], [x, x, x, x, x]], 
	// [[x, x, x, x, x], [x, x, x, x, x]], 
	// [[x, x, x, x, x], [x, x, x, x, x]]

	if (m_shape.dims() == 0) {
		throw std::runtime_error("Can not index tensor of zero dimensions");
		return std::unique_ptr<Tensor::Impl>(new Tensor::CPUImpl(*this));
	}

	if (m_shape.dims() == 1 && m_shape[0] == 1) {
		throw std::runtime_error("Can not index tensor of one dimension and one element");
		return std::unique_ptr<Tensor::Impl>(new Tensor::CPUImpl(*this));
	}

	Tensor::Shape newShape = m_shape.slice(1, m_shape.dims());
	Tensor::CPUImpl* results = new Tensor::CPUImpl(newShape);

	size_t offset = newShape.totalSize() * idx;
	for (size_t i = 0; i < newShape.totalSize(); i++) {
		results->m_data[i] = m_data[i + offset];
	}

	return std::unique_ptr<Tensor::Impl>(results);
}

void Tensor::CPUImpl::set(const std::vector<size_t>& position, const Tensor::Impl& other) {
	if (position.size() > m_shape.dims()) {
		throw std::runtime_error("Index specifies more dimensions than tensor has");
		return;
	}
	
	Tensor::Shape shapeOfBlock = m_shape.slice(position.size(), m_shape.dims());
	const Tensor::CPUImpl* o = dynamic_cast<const Tensor::CPUImpl*>(&other);

	if (!o) return;

	if (o->m_shape != shapeOfBlock) {
		throw std::runtime_error("Shape mismatch in assignment: cannot assign tensor of shape " +
			other.shape().toString() + " to tensor of shape " + this->shape().toString() +
			" with the " + std::to_string(shapeOfBlock.dims()) + " first dimensions being indexed");
		return;
	}

	size_t numElementsChanged = shapeOfBlock.totalSize();
	assert(o->m_data.size() == numElementsChanged && "assumed size of assignment tensor is wrong");

	size_t numBlocks = 1;
	for (size_t dimSize : position) {
		if (numBlocks == 0) numBlocks = 1;
		numBlocks *= dimSize;
	}

	size_t offset = numElementsChanged * numBlocks;
	for (size_t i = 0; i < numElementsChanged; i++) {
		m_data[i + offset] = o->m_data[i];
	}
}

bool Tensor::CPUImpl::compare(const std::vector<size_t>& position, const Tensor::Impl& other) const {
	const Tensor::CPUImpl* o = dynamic_cast<const Tensor::CPUImpl*>(&other);
	if (!o) return false;
	Tensor::Shape blockShape = m_shape.slice(position.size(), m_shape.dims());
	if (blockShape != o->m_shape) {
		return false;
	}

	// Calculate block size
	size_t blockSize = blockShape.totalSize();

	// Calulate number of blocks before this block
	size_t numBlocks = 1;
	for (size_t dimSize : position) {
		if (numBlocks == 0) numBlocks = 1;
		numBlocks *= dimSize;
	}

	size_t offset = blockSize * numBlocks;
	for (size_t i = 0; i < blockSize; ++i) {
		if (m_data[offset + i] != o->m_data[i]) return false;
	}

	return true;
}

bool Tensor::CPUImpl::compare(const std::vector<size_t>& position, const Tensor::Impl& other, const std::vector<size_t>& otherPosition) const {
    const Tensor::CPUImpl* o = dynamic_cast<const Tensor::CPUImpl*>(&other);
    if (!o) return false;
	// Get shapes of the blocks
	Tensor::Shape thisBlockShape = m_shape.slice(position.size(), m_shape.dims());
	Tensor::Shape otherBlockShape = o->m_shape.slice(otherPosition.size(), o->m_shape.dims());

	if (thisBlockShape != otherBlockShape) {
		return false;
	}

	// Calculate block size same for both
	size_t blockSize = thisBlockShape.totalSize();

	// Calculate offsets for both tensors
	size_t thisNumBlocks = 1;
	for (size_t dimSize : position) {
		if (thisNumBlocks == 0) thisNumBlocks = 1;
		thisNumBlocks *= dimSize;
	}

	size_t otherNumBlocks = 1;
	for (size_t dimSize : otherPosition) {
		if (otherNumBlocks == 0) otherNumBlocks = 1;
		otherNumBlocks *= dimSize;
	}

	size_t thisOffset = blockSize * thisNumBlocks;
	size_t otherOffset = blockSize * otherNumBlocks;

	// Compare the blocks element by element
	for (size_t i = 0; i < blockSize; i++) {
		if (m_data[thisOffset + i] != o->m_data[otherOffset + i]) {
			return false;
		}
	}

	return true;
}

bool Tensor::CPUImpl::operator==(const Tensor::Impl& other) const {
	if (m_shape != other.shape()) return false;

	const Tensor::CPUImpl* o = dynamic_cast<const Tensor::CPUImpl*>(&other);

	for (size_t i = 0; i < m_data.size(); i++) {
		if (m_data[i] != o->m_data[i]) {
			return false;
		}
	}

	return true;
}

bool Tensor::CPUImpl::operator!=(const Tensor::Impl& other) const {
	return !operator==(other);
}


///////////////////////////////////////////
// Element wise binary tensor operations //
///////////////////////////////////////////

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::add(const Tensor::Impl& other) const {
	Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape.toVector());

	const Tensor::CPUImpl* o = dynamic_cast<const Tensor::CPUImpl*>(&other);

	for (size_t i = 0; i < m_data.size(); i++) {
		results->m_data[i] = m_data[i] + o->m_data[i];
 	}

	return std::unique_ptr<Tensor::Impl>(results);
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::sub(const Tensor::Impl& other) const {
	Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape.toVector());

	const Tensor::CPUImpl* o = dynamic_cast<const Tensor::CPUImpl*>(&other);

	for (size_t i = 0; i < m_data.size(); i++) {
		results->m_data[i] = m_data[i] - o->m_data[i];
 	}

	return std::unique_ptr<Tensor::Impl>(results);
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::mul(const Tensor::Impl& other) const {
	Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape.toVector());

	const Tensor::CPUImpl* o = dynamic_cast<const Tensor::CPUImpl*>(&other);

	for (size_t i = 0; i < m_data.size(); i++) {
		results->m_data[i] = m_data[i] * o->m_data[i];
 	}

	return std::unique_ptr<Tensor::Impl>(results);
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::div(const Tensor::Impl& other) const {
	Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape.toVector());

	const Tensor::CPUImpl* o = dynamic_cast<const Tensor::CPUImpl*>(&other);

	for (size_t i = 0; i < m_data.size(); i++) {
		results->m_data[i] = m_data[i] / o->m_data[i];
 	}

	return std::unique_ptr<Tensor::Impl>(results);
}


//////////////////////////////////////////////////
// Element wise binary tensor-scalar operations //
//////////////////////////////////////////////////


std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::addScalar(const Tensor::Impl& other) const {
	Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape.toVector());

	float scalar = other.toVector()[0];

	for (size_t i = 0; i < m_data.size(); i++) {
		results->m_data[i] = m_data[i] + scalar;
 	}

	return std::unique_ptr<Tensor::Impl>(results);
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::subScalar(const Tensor::Impl& other) const {
	Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape.toVector());

	float scalar = other.toVector()[0];

	for (size_t i = 0; i < m_data.size(); i++) {
		results->m_data[i] = m_data[i] - scalar;
 	}

	return std::unique_ptr<Tensor::Impl>(results);
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::mulScalar(const Tensor::Impl& other) const {
	Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape.toVector());

	float scalar = other.toVector()[0];

	for (size_t i = 0; i < m_data.size(); i++) {
		results->m_data[i] = m_data[i] * scalar;
 	}

	return std::unique_ptr<Tensor::Impl>(results);
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::divScalar(const Tensor::Impl& other) const {
	Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape.toVector());

	float scalar = other.toVector()[0];

	for (size_t i = 0; i < m_data.size(); i++) {
		results->m_data[i] = m_data[i] / scalar;
 	}

	return std::unique_ptr<Tensor::Impl>(results);
}

