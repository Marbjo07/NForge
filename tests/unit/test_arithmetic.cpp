#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "nforge/nforge.h"
#include "utils.h"

TEST_CASE("Arithmetic ops behave identically on all backends", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({5}, 4.0f, backend);
		Tensor b({5}, 1.5f, backend);

		Tensor add = a + b;
		Tensor sub = a - b;
		Tensor mul = a * b;
		Tensor div = a / b;

		REQUIRE(add == Tensor({5}, 5.5f, backend));
		REQUIRE(sub == Tensor({5}, 2.5f, backend));
		REQUIRE(mul == Tensor({5}, 6.0f, backend));
		REQUIRE(div == Tensor({5}, 4.0f / 1.5f, backend));
	}
}

TEST_CASE("Scalar broadcasting works on all backends", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({4}, 2.0f, backend);
		Tensor s(3.0f, backend);

		Tensor x = a + s;
		Tensor y = s + a;
		Tensor z = a * s;

		REQUIRE(x == Tensor({4}, 5.0f, backend));
		REQUIRE(y == Tensor({4}, 5.0f, backend));
		REQUIRE(z == Tensor({4}, 6.0f, backend));
	}
}

TEST_CASE("2D arithmetic consistency across backends", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	auto rows = GENERATE(1ul, 3ul, 7ul);
	auto cols = GENERATE(1ul, 5ul, 11ul);

	DYNAMIC_SECTION("Backend=" << getBackendString(backend) << " rows=" << rows
	                           << " cols=" << cols) {
		Tensor A({rows, cols}, 2.0f, backend);
		Tensor B({rows, cols}, 0.5f, backend);

		Tensor C = A + B;
		Tensor D = A * B;

		REQUIRE(C == Tensor({rows, cols}, 2.5f, backend));
		REQUIRE(D == Tensor({rows, cols}, 1.0f, backend));
	}
}

TEST_CASE("In-place addition operator", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3}, 3.0f, backend);
		Tensor b({3}, 2.0f, backend);

		a += b;

		REQUIRE(a == Tensor({3}, 5.0f, backend));
	}
}

TEST_CASE("In-place subtraction operator", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3}, 3.0f, backend);
		Tensor b({3}, 2.0f, backend);

		a -= b;

		REQUIRE(a == Tensor({3}, 1.0f, backend));
	}
}

TEST_CASE("In-place multiplication operator", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3}, 3.0f, backend);
		Tensor b({3}, 2.0f, backend);

		a *= b;

		REQUIRE(a == Tensor({3}, 6.0f, backend));
	}
}

TEST_CASE("In-place division operator", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3}, 3.0f, backend);
		Tensor b({3}, 2.0f, backend);

		a /= b;

		REQUIRE(a == Tensor({3}, 1.5f, backend));
	}
}