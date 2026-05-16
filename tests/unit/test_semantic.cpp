#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "nforge/nforge.h"
#include "ops/semantic/semantic.h"

TEST_CASE("Tensor vs Tensor", "[Semantic]") {
	Tensor a({3}, 4.0f, Backend::CPU), b({3}, 4.0f, Backend::CPU);

	auto ctx = semantic::validateBinaryOperation(a, b);

	REQUIRE(ctx.lhs.offset == 0);
	REQUIRE(ctx.rhs.offset == 0);
	REQUIRE(ctx.out.rank == 1);
	REQUIRE(ctx.out.shape[0] == 3);
}

TEST_CASE("Tensor view vs Tensor view", "[Semantic]") {
	Tensor a({9, 8}, 4.0f, Backend::CPU), b({11, 8}, 4.0f, Backend::CPU);

	Tensor::View x = a[4];
	Tensor::View y = b[9];

	auto ctx = semantic::validateBinaryOperation(x, y);

	REQUIRE(ctx.lhs.offset == 8 * 4);
	REQUIRE(ctx.rhs.offset == 8 * 9);
	REQUIRE(ctx.out.rank == 1);
	REQUIRE(ctx.out.shape[0] == 8);
}

TEST_CASE("Tensor vs View", "[Semantic]") {
	Tensor a({9, 8}, 4.0f, Backend::CPU);
	Tensor b({8}, 1.0f, Backend::CPU);

	Tensor::View v = a[4];

	auto ctx = semantic::validateBinaryOperation(b, v);

	REQUIRE(ctx.lhs.offset == 0);
	REQUIRE(ctx.rhs.offset == 8 * 4);
	REQUIRE(ctx.out.rank == 1);
	REQUIRE(ctx.out.shape[0] == 8);
}

TEST_CASE("View vs Tensor", "[Semantic]") {
	Tensor a({9, 8}, 4.0f, Backend::CPU);
	Tensor b({8}, 1.0f, Backend::CPU);

	Tensor::View v = a[6];

	auto ctx = semantic::validateBinaryOperation(v, b);

	REQUIRE(ctx.lhs.offset == 8 * 6);
	REQUIRE(ctx.rhs.offset == 0);
	REQUIRE(ctx.out.rank == 1);
	REQUIRE(ctx.out.shape[0] == 8);
}

TEST_CASE("Scalar vs Tensor shape", "[Semantic]") {
	Tensor a({3, 4}, 1.0f, Backend::CPU);
	Tensor b({1}, 2.0f, Backend::CPU);

	auto ctx = semantic::validateBinaryOperation(a, b);

	REQUIRE(ctx.lhs.offset == 0);
	REQUIRE(ctx.rhs.offset == 0);
	REQUIRE(ctx.out.rank == 2);
	REQUIRE(ctx.out.shape[0] == 3);
	REQUIRE(ctx.out.shape[1] == 4);

	REQUIRE(ctx.rhs.strides[0] == 0);
	REQUIRE(ctx.rhs.strides[1] == 0);
}

TEST_CASE("Scalar vs Tensor view", "[Semantic]") {
	Tensor a({1}, 2.0f, Backend::CPU);
	Tensor b({3, 8, 4}, 1.0f, Backend::CPU);

	Tensor::View v = b[2];

	auto ctx = semantic::validateBinaryOperation(a, v);

	REQUIRE(ctx.lhs.offset == 0);
	REQUIRE(ctx.rhs.offset == 2 * 8 * 4);
	REQUIRE(ctx.out.rank == 2);
	REQUIRE(ctx.out.shape[0] == 8);
	REQUIRE(ctx.out.shape[1] == 4);

	REQUIRE(ctx.lhs.strides[0] == 0);
	REQUIRE(ctx.lhs.strides[1] == 0);
}

TEST_CASE("Broadcast (3,1) and (1,4) -> (3,4)", "[Semantic]") {
	Tensor a({3, 1}, 1.0f, Backend::CPU);
	Tensor b({1, 4}, 1.0f, Backend::CPU);

	auto ctx = semantic::validateBinaryOperation(a, b);

	REQUIRE(ctx.out.rank == 2);
	REQUIRE(ctx.out.shape[0] == 3);
	REQUIRE(ctx.out.shape[1] == 4);

	REQUIRE(ctx.lhs.strides[1] == 0);
	REQUIRE(ctx.rhs.strides[0] == 0);
}

TEST_CASE("Single element vs Tensor broadcasts", "[Semantic]") {
	Tensor a({1, 1}, 1.0f, Backend::CPU);
	Tensor b({3, 4}, 1.0f, Backend::CPU);

	auto ctx = semantic::validateBinaryOperation(a, b);

	REQUIRE(ctx.out.shape[0] == 3);
	REQUIRE(ctx.out.shape[1] == 4);
	REQUIRE(ctx.lhs.strides[0] == 0);
	REQUIRE(ctx.lhs.strides[1] == 0);
}

#ifdef NFORGE_WITH_CUDA
TEST_CASE("Throw on tensor device mismatch", "[Semantic]") {
	Tensor a({3}, 4.0f, Backend::CPU), b({3}, 4.0f, Backend::CUDA);

	REQUIRE_THROWS_AS(semantic::validateBinaryOperation(a, b), std::runtime_error);
	CHECK_THROWS_WITH(semantic::validateBinaryOperation(a, b),
	                  Catch::Matchers::ContainsSubstring("different devices"));
}

TEST_CASE("Throw on tensor view device mismatch", "[Semantic]") {
	Tensor a({3}, 4.0f, Backend::CPU), b({3}, 4.0f, Backend::CUDA);

	Tensor::View x = a[0];
	Tensor::View y = b[0];

	REQUIRE_THROWS_AS(semantic::validateBinaryOperation(x, y), std::runtime_error);
	CHECK_THROWS_WITH(semantic::validateBinaryOperation(x, y),
	                  Catch::Matchers::ContainsSubstring("different devices"));
}
#endif  // NFORGE_WITH_CUDA

TEST_CASE("Reduction operation", "[Semantic]") {
	Tensor a({1, 3, 5}, 1.0f, Backend::CPU);
	size_t dim = 1;

	// Shape = {1, 3, 5}
	// Block to reduce = {3, 5}
	// Resulting shape = {1}

	auto ctx = semantic::validateReduction(a, dim);

	REQUIRE(ctx.out.shape[0] == 1);
	REQUIRE(ctx.block.shape[0] == 3);
	REQUIRE(ctx.block.shape[1] == 5);
}

TEST_CASE("Throw on invalid dim in reduction operation", "[Semantic]") {
	Tensor a({1, 3, 5}, 1.0f, Backend::CPU);

	REQUIRE_THROWS(semantic::validateReduction(a, -1));
	REQUIRE_THROWS(semantic::validateReduction(a, 4));
}
