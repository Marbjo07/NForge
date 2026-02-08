#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "nforge/core/tensor.h"
#include "ops/semantic/semantic.h"

TEST_CASE("Tensor vs Tensor", "[semantic]") {
    Tensor a({3}, 4.0f, Backend::CPU), b({3}, 4.0f, Backend::CPU);

    auto ctx = semantic::validateBinaryOperation(a, b);

    REQUIRE(ctx.lhsOffset == 0);
    REQUIRE(ctx.rhsOffset == 0);
    REQUIRE(ctx.count == 3);
}

TEST_CASE("Tensor view vs Tensor view", "[semantic]") {
    Tensor a({9, 8}, 4.0f, Backend::CPU), b({11, 8}, 4.0f, Backend::CPU);

    Tensor::View x = a[4];
    Tensor::View y = b[9];

    auto ctx = semantic::validateBinaryOperation(x, y);

    REQUIRE(ctx.lhsOffset == 8 * (4));
    REQUIRE(ctx.rhsOffset == 8 * (9));
    REQUIRE(ctx.count == 8);
}

TEST_CASE("Tensor vs View", "[semantic]") {
    Tensor a({9,8}, 4.0f, Backend::CPU);
    Tensor b({8}, 1.0f, Backend::CPU);

    Tensor::View v = a[4];

    auto ctx = semantic::validateBinaryOperation(b, v);

    REQUIRE(ctx.lhsOffset == 0);
    REQUIRE(ctx.rhsOffset == 8 * 4);
    REQUIRE(ctx.count == 8);
    REQUIRE(ctx.shapeMatch == semantic::ShapeMatch::Equal);
}

TEST_CASE("View vs Tensor", "[semantic]") {
    Tensor a({9,8}, 4.0f, Backend::CPU);
    Tensor b({8}, 1.0f, Backend::CPU);

    Tensor::View v = a[6];

    auto ctx = semantic::validateBinaryOperation(v, b);

    REQUIRE(ctx.lhsOffset == 8 * 6);
    REQUIRE(ctx.rhsOffset == 0);
    REQUIRE(ctx.count == 8);
    REQUIRE(ctx.shapeMatch == semantic::ShapeMatch::Equal);
}

TEST_CASE("Scalar vs Tensor shape", "[semantic]") {
    Tensor a({3,4}, 1.0f, Backend::CPU);
    Tensor b({1}, 2.0f, Backend::CPU);

    auto ctx = semantic::validateBinaryOperation(a, b);

    REQUIRE(ctx.lhsOffset == 0);
    REQUIRE(ctx.rhsOffset == 0);
    REQUIRE(ctx.count == 12);
    REQUIRE(ctx.shapeMatch == semantic::ShapeMatch::ScalarRhs);
}

TEST_CASE("Scalar vs Tensor view", "[semantic]") {
    Tensor a({1}, 2.0f, Backend::CPU);
    Tensor b({3, 8, 4}, 1.0f, Backend::CPU);

    Tensor::View v = b[2];

    auto ctx = semantic::validateBinaryOperation(a, v);

    REQUIRE(ctx.lhsOffset == 0);
    REQUIRE(ctx.rhsOffset == 2 * 8 * 4);
    REQUIRE(ctx.count == 8 * 4);
    REQUIRE(ctx.shapeMatch == semantic::ShapeMatch::ScalarLhs);
}

TEST_CASE("Flat vs Shaped returns EqualCount", "[semantic]") {
    Tensor a({12}, 1.0f, Backend::CPU);
    Tensor b({3,4}, 1.0f, Backend::CPU);

    auto ctx = semantic::validateBinaryOperation(a, b);

    REQUIRE(ctx.lhsOffset == 0);
    REQUIRE(ctx.rhsOffset == 0);
    REQUIRE(ctx.count == 12);    
    REQUIRE(ctx.shapeMatch == semantic::ShapeMatch::EqualCount);
}

TEST_CASE("Throw on tensor device mismatch", "[semantic]") {
    Tensor a({3}, 4.0f, Backend::CPU), b({3}, 4.0f, Backend::CUDA);

    REQUIRE_THROWS_AS(semantic::validateBinaryOperation(a, b), std::runtime_error);
    CHECK_THROWS_WITH(semantic::validateBinaryOperation(a, b), Catch::Matchers::ContainsSubstring("different devices"));
}

TEST_CASE("Throw on tensor view device mismatch", "[semantic]") {
    Tensor a({3}, 4.0f, Backend::CPU), b({3}, 4.0f, Backend::CUDA);

    Tensor::View x = a[0];
    Tensor::View y = b[0];

    REQUIRE_THROWS_AS(semantic::validateBinaryOperation(x, y), std::runtime_error);
    CHECK_THROWS_WITH(semantic::validateBinaryOperation(x, y), Catch::Matchers::ContainsSubstring("different devices"));
}