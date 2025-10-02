#include "nforge/tensor.h"
#include "nforge/tensor_impl_CPU.h"

#include <algorithm>
#include <random>

Tensor::CPUImpl::CPUImpl(const std::vector<size_t>& shape) 
	: m_shape(shape) {

	if (m_shape.empty()) {
		m_shape.push_back(1);
	}

	size_t size = 1;
	for (size_t dimSize : shape) {
		size *= dimSize;
	}

	m_data.assign(size, 0.0f);
}

Tensor::CPUImpl::CPUImpl(const std::vector<size_t>& shape, float value) 
	: m_shape(shape) {
	
	if (m_shape.empty()) {
		m_shape.push_back(1);
	}
	
	size_t size = 1;
	for (size_t dimSize : shape) {
		size *= dimSize;
	}

	m_data.assign(size, value);
}

Tensor::CPUImpl::~CPUImpl() {
	m_data.clear();
	m_data.shrink_to_fit();

	m_shape.clear();
	m_shape.shrink_to_fit();
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

	std::vector<size_t> numElementsInDimsCurAndBelow(m_shape.size(), 1);
	for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; i--) {
		numElementsInDimsCurAndBelow[i] *= m_shape[i];
		if (i != static_cast<int>(m_shape.size()) - 1) {
			numElementsInDimsCurAndBelow[i] *= numElementsInDimsCurAndBelow[i + 1];
		}
	}

	for (size_t i = 0; i < m_data.size(); i++) {
		std::cout << m_data[i] << " ";
		for (size_t j = 0; j < m_shape.size(); j++) {
			if (i % numElementsInDimsCurAndBelow[j] == numElementsInDimsCurAndBelow[j] - 1) {
				std::cout << "\n";
			}
		}
	}

	std::cout << "Shape: " << getShapeAsString() << "\n";
	std::cout << "====================\n";
}

void Tensor::CPUImpl::print(const std::vector<size_t>& position) const {
	std::cout << "====================\n";
	std::cout << "Tensor[CPU], Data:\n";

	std::vector<size_t> numElementsInDimsCurAndBelow(m_shape.size(), 1);
	for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; i--) {
		numElementsInDimsCurAndBelow[i] *= m_shape[i];
		if (i != static_cast<int>(m_shape.size()) - 1) {
			numElementsInDimsCurAndBelow[i] *= numElementsInDimsCurAndBelow[i + 1];
		}
	}

	size_t blockSize = 1;
	for (int i = position.size(); i < m_shape.size(); i++) {
		blockSize *= m_shape[i];
	}

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
		for (size_t j = 0; j < m_shape.size(); j++) {
			if (i % numElementsInDimsCurAndBelow[j] == numElementsInDimsCurAndBelow[j] - 1) {
				std::cout << "\n";
			}
		}
	}
	std::cout << "\n";

	std::string shapeAsString = "{ ";
	for (size_t i = position.size(); i < m_shape.size(); i++) {
		shapeAsString += std::to_string(m_shape[i]) + " ";
	}
	if (position.size() == m_shape.size()) shapeAsString += "1 ";
	shapeAsString += "}";

	std::cout << "Shape: " << shapeAsString << "\n";
	std::cout << "====================\n";
}

std::string Tensor::CPUImpl::getShapeAsString() const {
	std::string out;

	out += "{ ";
	for (size_t dimSize : m_shape) {
		out += std::to_string(dimSize) + " ";
	}
	out += "}";

	return out;
}

std::string Tensor::CPUImpl::getDataAsString() const {
	std::string out;

	out += "{ ";
	for (float element : m_data) {
		out += std::to_string(element) + " ";
	}
	out += "}";

	return out;
}

size_t Tensor::CPUImpl::getNumberOfElements() const {
	size_t numImpliedByShape = 1;
	for (size_t dimSize : m_shape) {
		numImpliedByShape *= dimSize;
	}

	size_t numInContainer = m_data.size();

	assert(numImpliedByShape == numInContainer);

	return numInContainer;
}

std::vector<float> Tensor::CPUImpl::getAsVector() const {
	return m_data;
}

std::vector<size_t> Tensor::CPUImpl::getShape() const {
	return m_shape;
}

const float* Tensor::CPUImpl::getDataPointer() const {
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

	if (m_shape.size() == 0) {
		throw std::runtime_error("Can not index tensor of zero dimensions");
		return std::unique_ptr<Tensor::Impl>(new Tensor::CPUImpl(*this));
	}

	if (m_shape.size() == 1 && m_shape[0] == 1) {
		throw std::runtime_error("Can not index tensor of one dimension and one element");
		return std::unique_ptr<Tensor::Impl>(new Tensor::CPUImpl(*this));
	}

	std::vector<size_t> newShape{m_shape.begin() + 1, m_shape.end()};

	Tensor::CPUImpl* results = new Tensor::CPUImpl(newShape);

	size_t numElementsInResults = m_data.size() / m_shape[0];
	size_t readOffset = numElementsInResults * idx;
	
	for (size_t i = 0; i < numElementsInResults; i++) {
		results->m_data[i] = m_data[i + readOffset];
	}

	return std::unique_ptr<Tensor::Impl>(results);
}

void Tensor::CPUImpl::set(const std::vector<size_t>& position, const Tensor::Impl& other) {
	if (position.size() > m_shape.size()) {
		throw std::runtime_error("Index specifies more dimensions than tensor has");
		return;
	}

	std::vector<size_t> shapeOfBlock(m_shape.begin() + position.size(), m_shape.end());
	if (shapeOfBlock.empty()) {
		shapeOfBlock.push_back(1);
	}

	const Tensor::CPUImpl* o = dynamic_cast<const Tensor::CPUImpl*>(&other);


	// NOTE: TODO: this would be fixed with a tensor shape class
	//if (o->m_shape != shapeOfBlock) {
	//	throw std::runtime_error("Shape mismatch in assignment: cannot assign tensor of shape " +
	//		other.getShapeAsString() + " to tensor of shape " + this->getShapeAsString() +
	//		" with the " + std::to_string(shapeOfBlock.size()) + " first dimensions being indexed");
	//	return;
	//}

	size_t numElementsChanged = 1;
	for (size_t dimSize : shapeOfBlock) {
		numElementsChanged *= dimSize;
	}

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

	std::vector<size_t> blockShape(m_shape.begin() + position.size(), m_shape.end());
	if (blockShape.empty()) {
		blockShape.push_back(1);
	}

	if (blockShape != o->m_shape) {
		return false;
	}

	// Calculate block size
	size_t blockSize = 1;
	for (size_t dim : blockShape) {
		blockSize *= dim;
	}

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
    std::vector<size_t> thisBlockShape(m_shape.begin() + position.size(), m_shape.end());
    std::vector<size_t> otherBlockShape(o->m_shape.begin() + otherPosition.size(), o->m_shape.end());
    
    if (thisBlockShape.empty()) {
        thisBlockShape.push_back(1);
    }
    if (otherBlockShape.empty()) {
        otherBlockShape.push_back(1);
    }

    if (thisBlockShape != otherBlockShape) {
        return false;
    }

    // Calculate block size same for both
    size_t blockSize = 1;
    for (size_t dim : thisBlockShape) {
        blockSize *= dim;
    }

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

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::add(const Tensor::Impl& other) const {
	Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape);

	const Tensor::CPUImpl* o = dynamic_cast<const Tensor::CPUImpl*>(&other);

	for (size_t i = 0; i < m_data.size(); i++) {
		results->m_data[i] = m_data[i] + o->m_data[i];
 	}

	return std::unique_ptr<Tensor::Impl>(results);
}


std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::sub(const Tensor::Impl& other) const {
	Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape);

	const Tensor::CPUImpl* o = dynamic_cast<const Tensor::CPUImpl*>(&other);

	for (size_t i = 0; i < m_data.size(); i++) {
		results->m_data[i] = m_data[i] - o->m_data[i];
 	}

	return std::unique_ptr<Tensor::Impl>(results);
}


std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::mul(const Tensor::Impl& other) const {
	Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape);

	const Tensor::CPUImpl* o = dynamic_cast<const Tensor::CPUImpl*>(&other);

	for (size_t i = 0; i < m_data.size(); i++) {
		results->m_data[i] = m_data[i] * o->m_data[i];
 	}

	return std::unique_ptr<Tensor::Impl>(results);
}


std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::div(const Tensor::Impl& other) const {
	Tensor::CPUImpl* results = new Tensor::CPUImpl(this->m_shape);

	const Tensor::CPUImpl* o = dynamic_cast<const Tensor::CPUImpl*>(&other);

	for (size_t i = 0; i < m_data.size(); i++) {
		results->m_data[i] = m_data[i] / o->m_data[i];
 	}

	return std::unique_ptr<Tensor::Impl>(results);
}

std::unique_ptr<Tensor::Impl> Tensor::CPUImpl::pow(unsigned int exponent) const {
	auto result = std::make_unique<Tensor::CPUImpl>(this->m_shape, 1.0f);
	auto base = std::make_unique<Tensor::CPUImpl>(*this);

	while (exponent > 0) {
		if (exponent & 1) {
			for (size_t i = 0; i < result->m_data.size(); ++i) {
				result->m_data[i] *= base->m_data[i];
			}
		}
		exponent >>= 1;
		if (exponent > 0) {
			for (size_t i = 0; i < base->m_data.size(); ++i) {
				base->m_data[i] *= base->m_data[i];
			}
		}
	}

	return result;
}

bool Tensor::CPUImpl::operator==(const Tensor::Impl& other) const {
	if (m_shape != other.getShape()) return false;

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