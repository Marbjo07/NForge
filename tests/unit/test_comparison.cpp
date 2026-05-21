#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cmath>

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
			float lhsVal = lhs[i][j].copy().toVector()[0];
			float rhsVal = rhs[i][j].copy().toVector()[0];
			bool e = operand(lhsVal, rhsVal);

			expected[i][j] = e ? 1.0f : 0.0f;
		}
	}

	REQUIRE(result == expected);
}


template <typename A, typename B>
void testAllOperators(const A& lhs, const B& rhs, const std::string& desc = "") {
	auto suffix = desc.empty() ? "" : " (" + desc + ")";

	DYNAMIC_SECTION("<" + suffix) { checkComparison(lhs, rhs, std::less<>{}); }

	DYNAMIC_SECTION("<=" + suffix) { checkComparison(lhs, rhs, std::less_equal<>{}); }

	DYNAMIC_SECTION(">" + suffix) { checkComparison(lhs, rhs, std::greater<>{}); }

	DYNAMIC_SECTION(">=" + suffix) { checkComparison(lhs, rhs, std::greater_equal<>{}); }
}


Tensor randomIntegerTensor(const Tensor::Shape& shape, Backend backend) {
	Tensor t(shape, backend);
	t.fillRand();


	// TODO: refactor with vector init
	if (t.getShape().getNumDims() == 2) {
		for (size_t i = 0; i < t.getShape().getDim(0); i++) {
			for (size_t j = 0; j < t.getShape().getDim(1); j++) {
				float e = t[i][j].copy().toVector()[0];
				t[i][j] = std::round(e * 10);  // Scale and convert to int
			}
		}
	}

	// TODO: refactor with vector init
	if (t.getShape().getNumDims() == 3) {
		for (size_t i = 0; i < t.getShape().getDim(0); i++) {
			for (size_t j = 0; j < t.getShape().getDim(1); j++) {
				for (size_t k = 0; k < t.getShape().getDim(2); k++) {
					float e = t[i][j][k].copy().toVector()[0];
					t[i][j][k] = std::round(e * 10);  // Scale and convert to int
				}
			}
		}
	}

	return t;
}


TEST_CASE("Comparison Operators float", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor A({100, 100}, backend), B({100, 100}, backend);
		Tensor X({1, 100, 100}, backend), Y({1, 100, 100}, backend);
		auto xView = X[0];
		auto yView = Y[0];

		A.fillRand();
		B.fillRand();
		X.fillRand();
		Y.fillRand();

		testAllOperators(A, B, "Tensor-Tensor");
		testAllOperators(xView, yView, "View-View");
		testAllOperators(xView, B, "View-Tensor");
		testAllOperators(A, yView, "Tensor-View");
	}
}


TEST_CASE("Comparison Operators int", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor A = randomIntegerTensor({10, 10}, backend);
		Tensor B = randomIntegerTensor({10, 10}, backend);

		Tensor X = randomIntegerTensor({1, 10, 10}, backend);
		Tensor Y = randomIntegerTensor({1, 10, 10}, backend);
		auto xView = X[0];
		auto yView = Y[0];

		testAllOperators(A, B, "Tensor-Tensor");
		testAllOperators(xView, yView, "View-View");
		testAllOperators(xView, B, "View-Tensor");
		testAllOperators(A, yView, "Tensor-View");
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