#include "nforge/core/tensor.h"

#include "nforge/backend/cpu/tensor_impl_CPU.h"
#include "nforge/core/tensor_view.h"
#include "ops/semantic/semantic.h"

Tensor::Tensor(const Tensor::Shape& shape, Backend backend)
    : m_backend(backend) {
    if (backend == Backend::CPU) {
        m_impl = std::make_unique<Tensor::CPUImpl>(shape);
    } else {
        // TODO: should assert to false, but it's currently necessary to
        // have a valid non-cpu tensor in testing
        std::cout << "backend not implemented! defaulting to cpu\n";
        m_impl = std::make_unique<Tensor::CPUImpl>(shape);
    }
}

Tensor::Tensor(const Tensor::Shape& shape, float value, Backend backend)
    : Tensor(shape, backend) {
    m_impl->fillAll(value);
}

Tensor::Tensor(float value, Backend backend)
    : Tensor(Tensor::Shape({1}), value, backend) {
}

Tensor::Tensor(const Tensor& other)
    : m_backend(other.m_backend), m_impl(other.m_impl->clone()) {
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

void Tensor::set(const std::vector<size_t>& position, const Tensor& rhs) {
    Tensor::View lhs = Tensor::View((Tensor&)*this, position);

    auto ctx = semantic::validateBinaryOperation(lhs, rhs);
    if (ctx.shapeMatch != semantic::ShapeMatch::Equal) {
        throw std::runtime_error("Can't set position on tensors with shape mismatch, " + lhs.getShape().toString() + " and " + rhs.getShape().toString());
    }

    m_impl->set(ctx.lhsOffset, rhs.m_impl.get(), ctx.rhsOffset, ctx.count);
}

void Tensor::set(const std::vector<size_t>& position, const Tensor::View& rhs) {
    Tensor::View lhs = Tensor::View((Tensor&)*this, position);

    auto ctx = semantic::validateBinaryOperation(lhs, rhs);
    if (ctx.shapeMatch != semantic::ShapeMatch::Equal) {
        throw std::runtime_error("Can't set position on tensors with shape mismatch, " + lhs.getShape().toString() + " and " + rhs.getShape().toString());
    }

    m_impl->set(ctx.lhsOffset, rhs.getParent().m_impl.get(), ctx.rhsOffset, ctx.count);
}

bool Tensor::compare(const Tensor& rhs) const {
    auto ctx = semantic::validateBinaryOperation(*this, rhs);
    if (ctx.shapeMatch != semantic::ShapeMatch::Equal) {
        return false;
    }

    return m_impl->compare(ctx.lhsOffset, rhs.m_impl.get(), ctx.rhsOffset, ctx.count);
}

bool Tensor::compare(const Tensor::View& rhs) const {
    auto ctx = semantic::validateBinaryOperation(*this, rhs);
    if (ctx.shapeMatch != semantic::ShapeMatch::Equal) {
        return false;
    }

    return m_impl->compare(ctx.lhsOffset, rhs.getParent().m_impl.get(), ctx.rhsOffset, ctx.count);
}

bool Tensor::compare(const std::vector<size_t>& position, const Tensor& rhs) const {
    Tensor::View lhs((Tensor&)*this, position);

    auto ctx = semantic::validateBinaryOperation(lhs, rhs);
    if (ctx.shapeMatch != semantic::ShapeMatch::Equal) {
        return false;
    }

    return m_impl->compare(ctx.lhsOffset, rhs.m_impl.get(), ctx.rhsOffset, ctx.count);
}

bool Tensor::compare(const std::vector<size_t>& position, const Tensor::View& rhs) const {
    Tensor::View lhs((Tensor&)*this, position);

    auto ctx = semantic::validateBinaryOperation(lhs, rhs);
    if (ctx.shapeMatch != semantic::ShapeMatch::Equal) {
        return false;
    }

    return m_impl->compare(ctx.lhsOffset, rhs.getParent().m_impl.get(), ctx.rhsOffset, ctx.count);
}

template <typename EqualOp, typename ScalarOp>
Tensor Tensor::applyBinaryOp(const Tensor& rhs, const std::string& opName, EqualOp equalOp, ScalarOp scalarOp) const {
    auto ctx = semantic::validateBinaryOperation(*this, rhs);

    std::unique_ptr<Tensor::Impl> results;
    switch (ctx.shapeMatch) {
        case semantic::ShapeMatch::Equal:
            results = (m_impl.get()->*equalOp)(ctx.lhsOffset, rhs.m_impl.get(), ctx.rhsOffset, ctx.count);
            break;

        case semantic::ShapeMatch::ScalarLhs:
            results = (rhs.m_impl.get()->*scalarOp)(ctx.rhsOffset, m_impl.get(), ctx.count);
            break;

        case semantic::ShapeMatch::ScalarRhs:
            results = (m_impl.get()->*scalarOp)(ctx.lhsOffset, rhs.m_impl.get(), ctx.count);
            break;

        default:
            throw std::runtime_error("Can't " + opName + " tensors of incompatible shapes " + getShape().toString() + " and " + rhs.getShape().toString());
    }

    return Tensor(std::move(results), m_backend);
}

Tensor Tensor::operator+(const Tensor& rhs) const {
    return applyBinaryOp(rhs, "add", &Tensor::Impl::add, &Tensor::Impl::addScalar);
}

Tensor Tensor::operator-(const Tensor& rhs) const {
    return applyBinaryOp(rhs, "sub", &Tensor::Impl::sub, &Tensor::Impl::subScalar);
}

Tensor Tensor::operator/(const Tensor& rhs) const {
    return applyBinaryOp(rhs, "mul", &Tensor::Impl::mul, &Tensor::Impl::mulScalar);
}

Tensor Tensor::operator*(const Tensor& rhs) const {
    return applyBinaryOp(rhs, "div", &Tensor::Impl::div, &Tensor::Impl::divScalar);
}

Tensor::View Tensor::operator[](size_t idx) const {
    Tensor::View results((Tensor&)*this, {idx});
    return results;
}

Tensor Tensor::operator=(const Tensor& rhs) {
    this->m_impl = rhs.m_impl->clone();
    this->m_backend = rhs.m_backend;

    return *this;
}

bool Tensor::operator==(const Tensor& rhs) const {
    return compare(rhs);
}

bool Tensor::operator==(const Tensor::View& rhs) const {
    return compare(rhs);
}

bool Tensor::operator!=(const Tensor& rhs) const {
    return !operator==(rhs);
}

bool Tensor::operator!=(const Tensor::View& rhs) const {
    return !operator==(rhs);
}