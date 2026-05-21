#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "nforge/nforge.h"
#include "utils.h"


template <typename A, typename B, typename Operand>
void checkComparison(const A& lhs, const B& rhs, const Operand& operand) {
	Tensor result = operand(lhs, rhs);

	Tensor::Shape shape = lhs.getShape();
	for (size_t i = 0; i < shape.getDim(0); i++) {
		for (size_t j = 0; j < shape.getDim(1); j++) {
			bool expected = (lhs[i][j].copy().toVector()[0]) < (rhs[i][j].copy().toVector()[0]);

			REQUIRE(result[i][j] == Tensor(expected, backend));
		}
	}
}

TEST_CASE("Comparison Operators Tensor", "[Tensor]") {
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

TEST_CASE("Comparison Operators View", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({10, 10}, backend), b({10, 10}, backend);
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
		Tensor a({10, 10}, backend), b({10, 10}, backend);
		auto aView = a[0];
		auto bView = b[0];

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
		Tensor a({10, 10}, backend), b({10, 10}, backend);
		auto aView = a[0];
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