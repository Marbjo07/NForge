#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "nforge/nforge.h"
#include "utils.h"


// TODO: Tensor::View and Tensor should have same interface
Backend getBackend(const Tensor& t) { return t.getBackend(); }
Backend getBackend(const Tensor::View& v) { return v.getParent().getBackend(); }


template <typename A, typename B, typename Operand>
void checkComparison(const A& lhs, const B& rhs, const Operand& operand) {
	Tensor result = operand(lhs, rhs);
	Backend backend = getBackend(lhs);
	Tensor::Shape shape = lhs.getShape();

	// Verify tests are valid
	REQUIRE(shape == rhs.getShape());
	REQUIRE(shape.getNumDims() == 2);

	Tensor expected(shape, backend);
	for (size_t i = 0; i < shape.getDim(0); i++) {
		for (size_t j = 0; j < shape.getDim(1); j++) {
			// TODO: refactor with scalar to float conversion
			bool e = operand(lhs[i][j].copy().toVector()[0], rhs[i][j].copy().toVector()[0]);
			expected[i][j] = e ? 1.0f : 0.0f;
		}
	}

	REQUIRE(result == expected);
}

TEST_CASE("Comparison Operators Tensor vs Tensor", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({10, 10}, backend), b({10, 10}, backend);
		auto aView = a[0];
		auto bView = b[0];

		a.fillRand();
		b.fillRand();

		SECTION("Tensor < Tensor") {
			checkComparison(a, b, [](const auto& lhs, const auto& rhs) { return lhs < rhs; });
		}

		SECTION("Tensor <= Tensor") {
			checkComparison(a, b, [](const auto& lhs, const auto& rhs) { return lhs <= rhs; });
		}

		SECTION("Tensor > Tensor") {
			checkComparison(a, b, [](const auto& lhs, const auto& rhs) { return lhs > rhs; });
		}

		SECTION("Tensor >= Tensor") {
			checkComparison(a, b, [](const auto& lhs, const auto& rhs) { return lhs >= rhs; });
		}
	}
}

TEST_CASE("Comparison Operators View vs View", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({1, 10, 10}, backend), b({1, 10, 10}, backend);
		auto aView = a[0];
		auto bView = b[0];

		a.fillRand();
		b.fillRand();

		SECTION("View < View") {
			checkComparison(aView, bView,
			                [](const auto& lhs, const auto& rhs) { return lhs < rhs; });
		}

		SECTION("View <= View") {
			checkComparison(aView, bView,
			                [](const auto& lhs, const auto& rhs) { return lhs <= rhs; });
		}

		SECTION("View > View") {
			checkComparison(aView, bView,
			                [](const auto& lhs, const auto& rhs) { return lhs > rhs; });
		}

		SECTION("View >= View") {
			checkComparison(aView, bView,
			                [](const auto& lhs, const auto& rhs) { return lhs >= rhs; });
		}
	}
}

TEST_CASE("Comparison Operators View vs Tensor", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({1, 10, 10}, backend), b({10, 10}, backend);
		auto aView = a[0];

		a.fillRand();
		b.fillRand();

		SECTION("View < Tensor") {
			checkComparison(aView, b, [](const auto& lhs, const auto& rhs) { return lhs < rhs; });
		}

		SECTION("View <= Tensor") {
			checkComparison(aView, b, [](const auto& lhs, const auto& rhs) { return lhs <= rhs; });
		}

		SECTION("View > Tensor") {
			checkComparison(aView, b, [](const auto& lhs, const auto& rhs) { return lhs > rhs; });
		}

		SECTION("View >= Tensor") {
			checkComparison(aView, b, [](const auto& lhs, const auto& rhs) { return lhs >= rhs; });
		}
	}
}

TEST_CASE("Comparison Operators Tensor vs View", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({10, 10}, backend), b({1, 10, 10}, backend);
		auto bView = b[0];

		a.fillRand();
		b.fillRand();

		SECTION("Tensor < View") {
			checkComparison(a, bView, [](const auto& lhs, const auto& rhs) { return lhs < rhs; });
		}

		SECTION("Tensor <= View") {
			checkComparison(a, bView, [](const auto& lhs, const auto& rhs) { return lhs <= rhs; });
		}

		SECTION("Tensor > View") {
			checkComparison(a, bView, [](const auto& lhs, const auto& rhs) { return lhs > rhs; });
		}

		SECTION("Tensor >= View") {
			checkComparison(a, bView, [](const auto& lhs, const auto& rhs) { return lhs >= rhs; });
		}
	}
}


TEST_CASE("Comparison Operators Incompatible Shapes", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({10, 10}, backend), b({5, 5}, backend);

		REQUIRE_THROWS(a < b);
		REQUIRE_THROWS(a <= b);
		REQUIRE_THROWS(a > b);
		REQUIRE_THROWS(a >= b);
	}
}

#ifdef NFORGE_ENABLE_CUDA
TEST_CASE("Comparison Operators Incompatible Backends", "[Tensor]") {
	Tensor a({10, 10}, Backend::CPU);
	Tensor b({10, 10}, Backend::CUDA);

	REQUIRE_THROWS(a < b);
	REQUIRE_THROWS(a <= b);
	REQUIRE_THROWS(a > b);
	REQUIRE_THROWS(a >= b);
}
#endif  // NFORGE_ENABLE_CUDA