#include "nforge/core/tensor.h"
#include "nforge/core/tensor_view.h"

#include "backend/cpu/tensor_impl_CPU.h"
#include "backend/cuda/tensor_impl_CUDA.h"

#include "ops/semantic/semantic.h"

#ifdef NFORGE_WITH_CUDA
constexpr bool cudaEnabled = true;
#else
constexpr bool cudaEnabled = false;
#endif


Tensor::Tensor(const Tensor::Shape& shape, Backend backend)
    : m_backend(backend) {
    switch (backend) {
        case (Backend::CPU): 
            m_impl = std::make_unique<Tensor::CPUImpl>(shape);
            break;
        case (Backend::CUDA):
            if constexpr (cudaEnabled) {
                m_impl = std::make_unique<Tensor::CUDAImpl>(shape);
            }
            else {
                std::cout << "CUDA backend not built!";
                m_impl = std::make_unique<Tensor::CPUImpl>(shape);
            }
            break;
        default:
            std::cout << "backend not implemented! defaulting to cpu\n";
            m_impl = std::make_unique<Tensor::CPUImpl>(shape);
            break;
        }
}

Tensor::Tensor(const std::initializer_list<size_t>& shape, Backend backend)
    : Tensor(Tensor::Shape(shape), backend) {}


Tensor::Tensor(const Tensor::Shape& shape, float value, Backend backend)
    : Tensor(shape, backend) {
    m_impl->fillAll(value);
}

Tensor::Tensor(const std::initializer_list<size_t>& shape, float value, Backend backend)
    : Tensor(Tensor::Shape(shape), value, backend) {}

Tensor::Tensor(float value, Backend backend)
    : Tensor(Tensor::Shape({1}), value, backend) {}

Tensor::Tensor(const Tensor& rhs)
    : m_backend(rhs.m_backend), m_impl(rhs.m_impl->clone()) {
}

Tensor::Tensor(std::unique_ptr<Tensor::Impl> impl, Backend backend)
    : m_impl(std::move(impl)), m_backend(backend) {
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

Tensor::Shape Tensor::getShape() const {
    return m_impl->getShape();
}

std::string Tensor::getBackendString() const {
    switch (m_backend) {
        case Backend::CPU:
            return "CPU";
        case Backend::CUDA:
            return "CUDA";
        default:
            return "UNKNOWN";
    }
}

Backend Tensor::getBackend() const {
    return m_backend;
}

std::string Tensor::getDataString() const {
    return m_impl->toString();
}

size_t Tensor::getNumElements() const {
    return m_impl->getNumElements();
}

std::vector<float> Tensor::toVector() const {
    return m_impl->toVector();
}

void Tensor::set(const std::vector<size_t>& position, const Tensor::View& rhs) {
    Tensor::View lhs = Tensor::View((Tensor&)*this, position);

    auto ctx = semantic::validateBinaryOperation(lhs, rhs);

    auto outShape = Tensor::Shape(ctx.out);

    if (outShape != lhs.getShape()) {
        throw std::invalid_argument("set(): rhs shape does not broadcast to lhs shape");
    }

    m_impl->set(ctx.lhs, rhs.getParent().m_impl.get(), ctx.rhs);
}

bool Tensor::compare(const Tensor::View& rhs) const {
    if (this->getShape() != rhs.getShape()) {
        return false;
    }

    auto ctx = semantic::validateBinaryOperation(*this, rhs);
    return m_impl->compare(ctx.lhs, rhs.getParent().m_impl.get(), ctx.rhs);
}

bool Tensor::compare(const std::vector<size_t>& position, const Tensor::View& rhs) const {
    Tensor::View lhs((Tensor&)*this, position);

    if (lhs.getShape() != rhs.getShape()) {
        return false;
    }

    auto ctx = semantic::validateBinaryOperation(lhs, rhs);
    return m_impl->compare(ctx.lhs, rhs.getParent().m_impl.get(), ctx.rhs);
}

template <typename BinaryOp>
Tensor Tensor::applyBinaryOp(const Tensor::View& rhs, const std::string& opName, BinaryOp op) const {
    auto ctx = semantic::validateBinaryOperation(*this, rhs);

    Tensor::Impl* rhsImpl = rhs.getParent().m_impl.get();
    auto result = (m_impl.get()->*op)(ctx.lhs, rhsImpl, ctx.rhs, ctx.out);

    return Tensor(std::move(result), m_backend);
}

Tensor Tensor::operator+(const Tensor::View& rhs) const {
    return applyBinaryOp(rhs, "add", &Tensor::Impl::add);
}

Tensor Tensor::operator-(const Tensor::View& rhs) const {
    return applyBinaryOp(rhs, "sub", &Tensor::Impl::sub);
}

Tensor Tensor::operator*(const Tensor::View& rhs) const {
    return applyBinaryOp(rhs, "mul", &Tensor::Impl::mul);
}

Tensor Tensor::operator/(const Tensor::View& rhs) const {
    return applyBinaryOp(rhs, "div", &Tensor::Impl::div);
}


Tensor Tensor::operator+(float scalar) const {
    return *this + Tensor(scalar, m_backend);
}
Tensor Tensor::operator-(float scalar) const {
    return *this - Tensor(scalar, m_backend);
}
Tensor Tensor::operator*(float scalar) const {
    return *this * Tensor(scalar, m_backend);
}
Tensor Tensor::operator/(float scalar) const {
    return *this / Tensor(scalar, m_backend);
}

Tensor operator+(float scalar, const Tensor& rhs) {
    return Tensor(scalar, rhs.m_backend) + rhs;
}

Tensor operator-(float scalar, const Tensor& rhs) {
    return Tensor(scalar, rhs.m_backend) - rhs;
}

Tensor operator*(float scalar, const Tensor& rhs) {
    return Tensor(scalar, rhs.m_backend) * rhs;
}

Tensor operator/(float scalar, const Tensor& rhs) {
    return Tensor(scalar, rhs.m_backend) / rhs;
}

template <typename BinaryOp>
void Tensor::applyInplaceBinaryOp(const Tensor::View& rhs, const std::string& opName, BinaryOp op) {
    auto ctx = semantic::validateInplaceBinaryOperation(*this, rhs);

    Tensor::Impl* rhsImpl = rhs.getParent().m_impl.get();
    
    (m_impl.get()->*op)(ctx.lhs, rhsImpl, ctx.rhs);
}

void Tensor::operator+=(const Tensor::View& rhs) {
    applyInplaceBinaryOp(rhs, "add", &Tensor::Impl::iadd);
}

void Tensor::operator-=(const Tensor::View& rhs) {
    applyInplaceBinaryOp(rhs, "sub", &Tensor::Impl::isub);
}

void Tensor::operator*=(const Tensor::View& rhs) {
    applyInplaceBinaryOp(rhs, "mul", &Tensor::Impl::imul);
}

void Tensor::operator/=(const Tensor::View& rhs) {
    applyInplaceBinaryOp(rhs, "div", &Tensor::Impl::idiv);
}

template <typename ReductionOp>
Tensor Tensor::applyReduction(const std::string& reductionName, size_t dim, ReductionOp op) const {
    auto ctx = semantic::validateReduction(*this, dim);

    auto result = (m_impl.get()->*op)(ctx.lhs, ctx.block, ctx.out);

    return Tensor(std::move(result), m_backend);
}

Tensor Tensor::mean(size_t dim) const {
    Tensor res = this->sum(dim);
    Tensor::Shape block = getShape().getSlice(dim, getShape().getNumDims());

    return res / block.getNumElements();
}

Tensor Tensor::sum(size_t dim) const {
    return applyReduction("sum", dim, &Tensor::Impl::sum);
}

Tensor Tensor::min(size_t dim) const {
    return applyReduction("min", dim, &Tensor::Impl::min);
}

Tensor Tensor::max(size_t dim) const {
    return applyReduction("max", dim, &Tensor::Impl::max);
}

Tensor Tensor::prod(size_t dim) const {
    return applyReduction("prod", dim, &Tensor::Impl::prod);
}

Tensor Tensor::norm() const {
    Tensor::View view(*this);
    TensorLayout layout = view.getLayout();

    auto result = m_impl->norm(layout);
    return Tensor(std::move(result), m_backend);
}


Tensor::View Tensor::operator[](size_t idx) const {
    auto ctx = semantic::validateIndexing(*this, idx);

    std::vector<size_t> position = {idx};
    Tensor& parent = const_cast<Tensor&>(*this);

    Tensor::View out(parent, position, ctx.out);
    return out;
}

Tensor::View Tensor::subsample(std::vector<size_t> strides) const {
    Tensor::View view(*this);
    return view.subsample(strides);
}

Tensor& Tensor::operator=(const Tensor& rhs) {
    this->m_impl = rhs.m_impl->clone();
    this->m_backend = rhs.m_backend;

    return *this;
}

Tensor& Tensor::operator=(const Tensor::View& rhs) {
    applyInplaceBinaryOp(rhs, "set", &Tensor::Impl::set);

    return *this;
}

Tensor& Tensor::operator=(float scalar) {
    if (!this->getShape().isScalar()) {
        throw std::runtime_error("Cannot assign float to a non-scalar tensor.");
    }

    *this = Tensor(this->getShape(), scalar, this->m_backend);

    return *this;
}

bool Tensor::operator==(const Tensor::View& rhs) const {
    return compare(rhs);
}

bool Tensor::operator!=(const Tensor::View& rhs) const {
    return !operator==(rhs);
}