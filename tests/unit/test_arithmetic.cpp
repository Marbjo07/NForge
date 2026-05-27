#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "nforge/nforge.h"
#include "utils.h"

TEST_CASE("Tensor-Tensor arithmetic", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({5}, 4.0f, backend);
		Tensor b({5}, 1.5f, backend);

		REQUIRE(a + b == Tensor({5}, 5.5f, backend));
		REQUIRE(a - b == Tensor({5}, 2.5f, backend));
		REQUIRE(a * b == Tensor({5}, 6.0f, backend));
		REQUIRE(a / b == Tensor({5}, 4.0f / 1.5f, backend));
	}
}

TEST_CASE("Tensor-View arithmetic", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({5}, 4.0f, backend);
		Tensor Y({1, 5}, 1.5f, backend);
		auto yView = Y[0];

		REQUIRE(a + yView == Tensor({5}, 5.5f, backend));
		REQUIRE(a - yView == Tensor({5}, 2.5f, backend));
		REQUIRE(a * yView == Tensor({5}, 6.0f, backend));
		REQUIRE(a / yView == Tensor({5}, 4.0f / 1.5f, backend));
	}
}

TEST_CASE("View-Tensor arithmetic", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor X({1, 5}, 4.0f, backend);
		Tensor b({5}, 1.5f, backend);
		auto xView = X[0];

		REQUIRE(xView + b == Tensor({5}, 5.5f, backend));
		REQUIRE(xView - b == Tensor({5}, 2.5f, backend));
		REQUIRE(xView * b == Tensor({5}, 6.0f, backend));
		REQUIRE(xView / b == Tensor({5}, 4.0f / 1.5f, backend));
	}
}

TEST_CASE("View-View arithmetic", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor X({1, 5}, 4.0f, backend);
		Tensor Y({1, 5}, 1.5f, backend);
		auto xView = X[0];
		auto yView = Y[0];

		REQUIRE(xView + yView == Tensor({5}, 5.5f, backend));
		REQUIRE(xView - yView == Tensor({5}, 2.5f, backend));
		REQUIRE(xView * yView == Tensor({5}, 6.0f, backend));
		REQUIRE(xView / yView == Tensor({5}, 4.0f / 1.5f, backend));
	}
}

TEST_CASE("Tensor-Scalar arithmetic", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({4}, 6.0f, backend);
		Tensor s(2.0f, backend);

		REQUIRE(a + s == Tensor({4}, 8.0f, backend));
		REQUIRE(a - s == Tensor({4}, 4.0f, backend));
		REQUIRE(a * s == Tensor({4}, 12.0f, backend));
		REQUIRE(a / s == Tensor({4}, 3.0f, backend));
	}
}

TEST_CASE("Scalar-Tensor arithmetic", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor s(6.0f, backend);
		Tensor a({4}, 2.0f, backend);

		REQUIRE(s + a == Tensor({4}, 8.0f, backend));
		REQUIRE(s - a == Tensor({4}, 4.0f, backend));
		REQUIRE(s * a == Tensor({4}, 12.0f, backend));
		REQUIRE(s / a == Tensor({4}, 3.0f, backend));
	}
}

TEST_CASE("Tensor-Vector arithmetic", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({2, 3}, 4.0f, backend);
		Tensor b({3}, 2.0f, backend);

		REQUIRE(a + b == Tensor({2, 3}, 6.0f, backend));
		REQUIRE(a - b == Tensor({2, 3}, 2.0f, backend));
		REQUIRE(a * b == Tensor({2, 3}, 8.0f, backend));
		REQUIRE(a / b == Tensor({2, 3}, 2.0f, backend));
	}
}

TEST_CASE("In-place Tensor-Tensor", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3}, 3.0f, backend);
		Tensor b({3}, 2.0f, backend);

		SECTION("+=") {
			a += b;
			REQUIRE(a == Tensor({3}, 5.0f, backend));
		}
		SECTION("-=") {
			a -= b;
			REQUIRE(a == Tensor({3}, 1.0f, backend));
		}
		SECTION("*=") {
			a *= b;
			REQUIRE(a == Tensor({3}, 6.0f, backend));
		}
		SECTION("/=") {
			a /= b;
			REQUIRE(a == Tensor({3}, 1.5f, backend));
		}
	}
}

TEST_CASE("In-place View-View", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor X({1, 3}, 3.0f, backend);
		Tensor Y({1, 3}, 2.0f, backend);
		auto xView = X[0];
		auto yView = Y[0];
		SECTION("+=") {
			xView += yView;
			REQUIRE(X == Tensor({1, 3}, 5.0f, backend));
		}
		SECTION("-=") {
			xView -= yView;
			REQUIRE(X == Tensor({1, 3}, 1.0f, backend));
		}
		SECTION("*=") {
			xView *= yView;
			REQUIRE(X == Tensor({1, 3}, 6.0f, backend));
		}
		SECTION("/=") {
			xView /= yView;
			REQUIRE(X == Tensor({1, 3}, 1.5f, backend));
		}
	}
}

TEST_CASE("In-place Tensor-View", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({2, 3}, 4.0f, backend);
		Tensor b({3}, 1.5f, backend);
		SECTION("+=") {
			a += b;
			REQUIRE(a == Tensor({2, 3}, 5.5f, backend));
		}
		SECTION("-=") {
			a -= b;
			REQUIRE(a == Tensor({2, 3}, 2.5f, backend));
		}
		SECTION("*=") {
			a *= b;
			REQUIRE(a == Tensor({2, 3}, 6.0f, backend));
		}
		SECTION("/=") {
			a /= b;
			REQUIRE(a == Tensor({2, 3}, 4.0f / 1.5f, backend));
		}
	}
}

TEST_CASE("Incompatible shapes throw", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3, 4}, backend);
		Tensor b({2, 5}, backend);

		REQUIRE_THROWS(a + b);
		REQUIRE_THROWS(a - b);
		REQUIRE_THROWS(a * b);
		REQUIRE_THROWS(a / b);


		REQUIRE_THROWS(a += b);
		REQUIRE_THROWS(a -= b);
		REQUIRE_THROWS(a *= b);
		REQUIRE_THROWS(a /= b);
	}
}

TEST_CASE("Incompatible broadcast dimensions throw", "[Tensor][Arithmetic]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3, 4}, backend);
		Tensor b({3}, backend);

		REQUIRE_THROWS(a + b);
		REQUIRE_THROWS(a - b);
		REQUIRE_THROWS(a * b);
		REQUIRE_THROWS(a / b);


		REQUIRE_THROWS(a += b);
		REQUIRE_THROWS(a -= b);
		REQUIRE_THROWS(a *= b);
		REQUIRE_THROWS(a /= b);
	}
}

#ifdef NFORGE_ENABLE_CUDA
TEST_CASE("Incompatible backends throw", "[Tensor][Arithmetic]") {
	Tensor a({5}, Backend::CPU);
	Tensor b({5}, Backend::CUDA);

	REQUIRE_THROWS(a + b);
	REQUIRE_THROWS(a - b);
	REQUIRE_THROWS(a * b);
	REQUIRE_THROWS(a / b);
}
#endif  // NFORGE_ENABLE_CUDA