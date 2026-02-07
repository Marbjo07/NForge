#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "nforge/core/tensor.h"
#include "ops/semantic/semantic.h"

TEST_CASE("Correct return values for tensor", "[semantic]") {
	Tensor a({3}, 4.0f, Backend::CPU), b({3}, 4.0f, Backend::CPU);

    auto ctx = nforge::semantic::validateBinaryOperation(a, b);

    REQUIRE(ctx.lhsOffset == 0);
    REQUIRE(ctx.rhsOffset == 0);
    REQUIRE(ctx.count == 3);
}

TEST_CASE("Correct return values for tensor view", "[semantic]") {
	Tensor a({9, 8}, 4.0f, Backend::CPU), b({11, 8}, 4.0f, Backend::CPU);
    
    Tensor::View x = a[4];
    Tensor::View y = b[9];

    auto ctx = nforge::semantic::validateBinaryOperation(x, y);

    REQUIRE(ctx.lhsOffset == 8 * (4));
    REQUIRE(ctx.rhsOffset == 8 * (9));
    REQUIRE(ctx.count == 8);
}


TEST_CASE("Throw on tensor device mismatch", "[semantic]") {
    Tensor a({3}, 4.0f, Backend::CPU), b({3}, 4.0f, Backend::CUDA);

    REQUIRE_THROWS_AS(nforge::semantic::validateBinaryOperation(a, b), std::runtime_error);
    CHECK_THROWS_WITH(nforge::semantic::validateBinaryOperation(a, b), Catch::Matchers::ContainsSubstring("different devices"));
}

TEST_CASE("Throw on tensor view device mismatch", "[semantic]") {
    Tensor a({3}, 4.0f, Backend::CPU), b({3}, 4.0f, Backend::CUDA);

    Tensor::View x = a[0];
    Tensor::View y = b[0];


    REQUIRE_THROWS_AS(nforge::semantic::validateBinaryOperation(x, y), std::runtime_error);
    CHECK_THROWS_WITH(nforge::semantic::validateBinaryOperation(x, y), Catch::Matchers::ContainsSubstring("different devices"));
}