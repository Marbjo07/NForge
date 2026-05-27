#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "nforge/nforge.h"
#include "utils.h"

TEST_CASE("View copy produces an independent Tensor", "[View]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor src({3, 4}, 7.0f, backend);

		auto view = src[1];  // view of row-1 (4 elements)
		Tensor copy = view.copy();

		// Dimensions match the viewed slice
		REQUIRE(copy.getShape() == Tensor::Shape({4}));

		// Values match
		REQUIRE(copy == Tensor({4}, 7.0f, backend));
	}
}

TEST_CASE("View copy is a deep copy", "[View]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor src({4}, 1.0f, backend);
		auto view = src[0];  // scalar view
		Tensor copy = view.copy();

		// Mutate the source and copy must be unaffected
		src = Tensor({4}, 99.0f, backend);
		REQUIRE(copy == Tensor(1.0f, backend));
	}
}

TEST_CASE("Copy of a 2D sub-view preserves values", "[View]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor src({3, 5}, 4.0f, backend);
		Tensor row2 = src[2].copy();

		REQUIRE(row2 == Tensor({5}, 4.0f, backend));
	}
}

TEST_CASE("2D View arithmetic consistency across backends", "[View][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	auto rows = GENERATE(2ul, 4ul, 8ul);
	auto cols = GENERATE(1ul, 5ul, 11ul);

	DYNAMIC_SECTION("Backend=" << getBackendString(backend) << " rows=" << rows
	                           << " cols=" << cols) {
		Tensor A({rows, cols}, 6.0f, backend);
		Tensor B({rows, cols}, 2.0f, backend);

		for (size_t i = 0; i < rows; i++) {
			Tensor sum = A[i] + B[i];
			Tensor diff = A[i] - B[i];
			Tensor prod = A[i] * B[i];
			Tensor quot = A[i] / B[i];

			REQUIRE(sum == Tensor({cols}, 8.0f, backend));
			REQUIRE(diff == Tensor({cols}, 4.0f, backend));
			REQUIRE(prod == Tensor({cols}, 12.0f, backend));
			REQUIRE(quot == Tensor({cols}, 3.0f, backend));
		}
	}
}

TEST_CASE("Mixed Tensor/View 2D parametric test", "[Tensor][View][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	auto rows = GENERATE(1ul, 3ul, 7ul);
	auto cols = GENERATE(1ul, 5ul, 11ul);

	DYNAMIC_SECTION("Backend=" << getBackendString(backend) << " rows=" << rows
	                           << " cols=" << cols) {
		Tensor A({rows, cols}, 4.0f, backend);
		Tensor vec({cols}, 1.0f, backend);

		for (size_t i = 0; i < rows; i++) {
			Tensor sum = A[i] + vec;   // view + tensor
			Tensor prod = vec * A[i];  // tensor * view

			REQUIRE(sum == Tensor({cols}, 5.0f, backend));
			REQUIRE(prod == Tensor({cols}, 4.0f, backend));
		}
	}
}

TEST_CASE("Chained view arithmetic expressions", "[View][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor A({2, 4}, 2.0f, backend);
		Tensor B({2, 4}, 3.0f, backend);
		Tensor C({2, 4}, 4.0f, backend);

		// (A[0] + B[0]) * C[0] => (2+3)*4 = 20
		Tensor result = (A[0] + B[0]) * C[0];

		REQUIRE(result == Tensor({4}, 20.0f, backend));

		// A, B, C should remain unchanged
		REQUIRE(A == Tensor({2, 4}, 2.0f, backend));
		REQUIRE(B == Tensor({2, 4}, 3.0f, backend));
		REQUIRE(C == Tensor({2, 4}, 4.0f, backend));
	}
}

TEST_CASE("Copy then arithmetic gives correct result", "[View][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor M({3, 4}, 5.0f, backend);

		Tensor row = M[1].copy();
		Tensor other({4}, 2.0f, backend);

		Tensor result = row + other;

		REQUIRE(result == Tensor({4}, 7.0f, backend));
	}
}

TEST_CASE("Arithmetic result does not alias the source view", "[View][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor A({2, 4}, 1.0f, backend);
		Tensor B({2, 4}, 2.0f, backend);

		Tensor sum = A[0] + B[0];  // should be 3.0

		// Mutate A - sum must stay 3.0
		A = Tensor({2, 4}, 99.0f, backend);

		REQUIRE(sum == Tensor({4}, 3.0f, backend));
	}
}