#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cmath>

#include "nforge/nforge.h"
#include "utils.h"

// TODO: Tensor::View and Tensor should have same interface
Backend getBackend(const Tensor& t) { return t.getBackend(); }
Backend getBackend(const Tensor::View& v) { return v.getParent().getBackend(); }


std::vector<float> getVector(const Tensor& t) { return t.toVector(); }
std::vector<float> getVector(const Tensor::View& t) { return t.copy().toVector(); }


template <typename A, typename B, typename Operand>
void checkComparison(const A& lhs, const B& rhs, const Operand& operand) {
	Tensor result = operand(lhs, rhs);
	Backend backend = getBackend(lhs);

	Tensor::Shape expectedShape;

	// Verify tests are valid
	bool lhsIsScalar = lhs.getShape().isScalar();
	bool rhsIsScalar = rhs.getShape().isScalar();

	if (lhsIsScalar || rhsIsScalar) {
		// get a non-scalar shape, if one is scalar
		if (lhsIsScalar)
			expectedShape = rhs.getShape();
		if (rhsIsScalar)
			expectedShape = lhs.getShape();
	} else {
		REQUIRE(lhs.getShape() == rhs.getShape());
		expectedShape = lhs.getShape();
	}

	REQUIRE(expectedShape.getNumDims() == 2);

	Tensor expected(expectedShape, backend);

	// calling toVector is expensive for cuda, cus cudaSync
	// copy to CPU and compare there to avoid this
	auto lhsVec = getVector(lhs);
	auto rhsVec = getVector(rhs);

	for (size_t i = 0; i < expectedShape.getDim(0); i++) {
		for (size_t j = 0; j < expectedShape.getDim(1); j++) {
			int lhsIdx = i * expectedShape.getDim(1) + j;
			int rhsIdx = i * expectedShape.getDim(1) + j;

			// check if scalar and adjust idx if so
			if (lhs.getShape().isScalar())
				lhsIdx = 0;
			if (rhs.getShape().isScalar())
				rhsIdx = 0;

			float lhsVal = lhsVec[lhsIdx];
			float rhsVal = rhsVec[rhsIdx];

			bool e = operand(lhsVal, rhsVal);

			expected[i][j] = e ? 1.0f : 0.0f;
		}
	}

	REQUIRE(result == expected);
}

template <typename A, typename B>
void testAllOperators(const A& lhs, const B& rhs, const std::string& desc) {
	auto suffix = " (" + desc + ")";

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
		Tensor A({20, 20}, backend), B({20, 20}, backend);
		Tensor X({1, 20, 20}, backend), Y({1, 20, 20}, backend);
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

TEST_CASE("Comparison Operators scalar broadcast", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({10, 10}, backend);
		Tensor b({1, 10, 10}, backend);
		auto view = b[0];

		// init middel of rand distribution
		Tensor scalar({1}, 0, backend);

		a.fillRand();
		b.fillRand();

		testAllOperators(a, scalar, "Tensor-Scalar");
		testAllOperators(scalar, a, "Scalar-Tensor");

		testAllOperators(view, scalar, "View-Scalar");
		testAllOperators(scalar, view, "Scalar-View");
	}
}

TEST_CASE("Comparison Operators scalar broadcast int", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a = randomIntegerTensor({10, 10}, backend);
		Tensor b = randomIntegerTensor({1, 10, 10}, backend);
		auto view = b[0];

		// init middel of rand distribution
		Tensor scalar({1}, 0, backend);

		testAllOperators(a, scalar, "Tensor-Scalar");
		testAllOperators(scalar, a, "Scalar-Tensor");

		testAllOperators(view, scalar, "View-Scalar");
		testAllOperators(scalar, view, "Scalar-View");
	}
}

template <typename Operand>
void testBroadcastComparison(const Tensor& a, const Tensor::View& b, const Operand& operand) {
	Tensor result = operand(a, b);
	Tensor expected(result.getShape(), result.getBackend());

	for (size_t i = 0; i < result.getShape().getDim(0); i++) {
		for (size_t j = 0; j < result.getShape().getDim(1); j++) {
			float aVal = a[i][j].copy().toVector()[0];
			float bVal = b[j].copy().toVector()[0];

			expected[i][j] = operand(aVal, bVal) ? 1.0f : 0.0f;
		}
	}

	REQUIRE(result == expected);
}


TEST_CASE("Comparison Operators non-scalar broadcast", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		size_t n = 10, m = 30;
		Tensor a({n, m}, backend);
		Tensor b({m}, backend);

		a.fillRand();
		b.fillRand();

		SECTION("<") { testBroadcastComparison(a, b, std::less<>()); }
		SECTION("<=") { testBroadcastComparison(a, b, std::less_equal<>()); }
		SECTION(">") { testBroadcastComparison(a, b, std::greater<>()); }
		SECTION(">=") { testBroadcastComparison(a, b, std::greater_equal<>()); }
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

TEST_CASE("isClose", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		SECTION("Identical values") {
			Tensor a({10, 10}, 3.14f, backend);
			Tensor b({10, 10}, 3.14f, backend);

			Tensor result = a.isClose(b);
			REQUIRE(result == Tensor({10, 10}, 1.0f, backend));
		}

		SECTION("Within tolerance") {
			Tensor a({10}, 100.0f, backend);
			Tensor b({10}, 100.0f, backend);

			for (size_t i = 0; i < 10; i++) {
				// TODO: in-place operators does not work for views?
				b[i] = b[i] + Tensor(1e-6f * (i + 1), backend);
			}

			Tensor result = a.isClose(b);
			REQUIRE(result == Tensor({10}, 1.0f, backend));
		}

		SECTION("Different values") {
			Tensor a({10}, 1.0f, backend);
			Tensor b({10}, 2.0f, backend);

			Tensor result = a.isClose(b);
			REQUIRE(result == Tensor({10}, 0.0f, backend));
		}

		SECTION("Self comparison") {
			Tensor a({4, 5}, 0.0f, backend);
			a.fillRand();

			Tensor closeResult = a.isClose(a);
			REQUIRE(closeResult == Tensor({4, 5}, 1.0f, backend));

			Tensor farResult = a.isClose(a + Tensor({4, 5}, 1.0f, backend));
			REQUIRE(farResult == Tensor({4, 5}, 0.0f, backend));
		}

		SECTION("Mixed case") {
			Tensor a({5}, backend);
			Tensor b({5}, backend);

			a[0] = 1.0f;
			b[0] = 1.0f;

			a[1] = 1.0f;
			b[1] = 1.0f + 1e-6f;

			a[2] = 1.0f;
			b[2] = 2.0f;

			a[3] = 100.0f;
			b[3] = 100.0f + 1e-6f;

			a[4] = 100.0f;
			b[4] = 101.0f;

			Tensor result = a.isClose(b);

			REQUIRE(result[0] == Tensor(1.0f, backend));
			REQUIRE(result[1] == Tensor(1.0f, backend));
			REQUIRE(result[2] == Tensor(0.0f, backend));
			REQUIRE(result[3] == Tensor(1.0f, backend));
			REQUIRE(result[4] == Tensor(0.0f, backend));
		}
	}
}

TEST_CASE("isClose custom tolerance", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3}, 1.0f, backend);
		Tensor b({3}, 1.0f, backend);

		b[0] = 1.0f + 1e-4f;
		b[1] = 1.0f + 1e-6f;
		b[2] = 1.0f + 1e-1f;

		SECTION("Default tolerance") {
			Tensor resultDefault = a.isClose(b);
			REQUIRE(resultDefault[0] == Tensor(0.0f, backend));
			REQUIRE(resultDefault[1] == Tensor(1.0f, backend));
			REQUIRE(resultDefault[2] == Tensor(0.0f, backend));
		}

		SECTION("Tight tolerance") {
			Tensor resultTight = a.isClose(b, 1e-7f);
			REQUIRE(resultTight == Tensor({3}, 0.0f, backend));
		}

		SECTION("Loose tolerance") {
			Tensor resultLoose = a.isClose(b, 1e-1f);
			REQUIRE(resultLoose == Tensor({3}, 1.0f, backend));
		}
	}
}

TEST_CASE("isClose edge cases", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		SECTION("Zero values") {
			Tensor a({4}, 0.0f, backend);
			Tensor b({4}, 0.0f, backend);

			a[0] = 0.0f;
			a[1] = 1e-6f;
			a[2] = 1e-4f;
			a[3] = -1e-6f;

			Tensor result = a.isClose(b);

			REQUIRE(result[0] == Tensor(1.0f, backend));
			REQUIRE(result[1] == Tensor(1.0f, backend));
			REQUIRE(result[2] == Tensor(0.0f, backend));
			REQUIRE(result[3] == Tensor(1.0f, backend));
		}

		SECTION("Absolute regime boundary") {
			Tensor a({4}, 0.0f, backend);
			Tensor b({4}, 0.0f, backend);
			a[0] = 1.01e-5f;
			a[1] = -1.01e-5f;

			a[2] = 0.99e-5f;
			a[3] = -0.99e-5f;

			Tensor result = a.isClose(b);
			REQUIRE(result[0] == Tensor(0.0f, backend));
			REQUIRE(result[1] == Tensor(0.0f, backend));
			REQUIRE(result[2] == Tensor(1.0f, backend));
			REQUIRE(result[3] == Tensor(1.0f, backend));
		}

		SECTION("Negative values") {
			Tensor a({3}, 0.0f, backend);
			Tensor b({3}, 0.0f, backend);

			a[0] = -1.5f;
			b[0] = -1.5f;

			a[1] = -100.0f;
			b[1] = -100.0f + 1e-6f;

			a[2] = -1.5f;
			b[2] = -2.0f;

			Tensor result = a.isClose(b);

			REQUIRE(result[0] == Tensor(1.0f, backend));
			REQUIRE(result[1] == Tensor(1.0f, backend));
			REQUIRE(result[2] == Tensor(0.0f, backend));
		}
	}
}

TEST_CASE("isClose boundary", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		float eps = 1e-5f;

		SECTION("Inside tolerance") {
			Tensor a({2}, 1.0f, backend);
			Tensor b({2}, 1.0f, backend);

			b[0] = 1.0f + eps * 0.5f;
			b[1] = 1.0f - eps * 0.5f;

			Tensor result = a.isClose(b);
			REQUIRE(result == Tensor({2}, 1.0f, backend));
		}

		SECTION("Inside tolerance near zero") {
			Tensor a({3}, 0.0f, backend);
			Tensor b({3}, 0.0f, backend);

			b[0] = 0.0f + eps * 0.5f;
			b[1] = 0.0f - eps * 0.5f;
			b[2] = 0.0f;

			Tensor result = a.isClose(b);
			REQUIRE(result == Tensor({3}, 1.0f, backend));
		}

		SECTION("Outside tolerance") {
			Tensor a({3}, 1.0f, backend);
			Tensor b({3}, 1.0f, backend);

			b[0] = 1.0f + eps * 10.0f;
			b[1] = 1.0f - eps * 10.0f;
			b[2] = 100.0f - eps * 10.0f;

			Tensor result = a.isClose(b);
			REQUIRE(result == Tensor({3}, 0.0f, backend));
		}

		SECTION("Relative regime") {
			Tensor a({2}, 1000.0f, backend);
			Tensor b({2}, 1000.0f, backend);

			b[0] = 1000.0f + 0.5f;
			b[1] = 1000.0f + 1e-6f;

			Tensor result = a.isClose(b);

			REQUIRE(result[0] == Tensor(0.0f, backend));
			REQUIRE(result[1] == Tensor(1.0f, backend));
		}
	}
}

TEST_CASE("isClose broadcasting scalar", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({5}, 1.0f, backend);
		Tensor scalar(1.0f + 1e-6f, backend);

		Tensor result = a.isClose(scalar);
		REQUIRE(result == Tensor({5}, 1.0f, backend));

		Tensor farScalar(100.0f, backend);
		Tensor resultFar = a.isClose(farScalar);
		REQUIRE(resultFar == Tensor({5}, 0.0f, backend));
	}
}

TEST_CASE("isClose broadcasting non scalar", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		size_t n = 5, m = 10;
		Tensor a({n, m}, 0.0f, backend);
		Tensor b({m}, 0.0f, backend);

		for (size_t i = 0; i < n; i++) {
			for (size_t j = 0; j < m; j++) {
				a[i][j] = static_cast<float>(i * m + j) + 1e-7f;
			}
		}
		for (size_t j = 0; j < m; j++) {
			b[j] = static_cast<float>(j);
		}

		Tensor result = a.isClose(b);
		REQUIRE(result.getShape() == Tensor::Shape({n, m}));

		REQUIRE(result[0] == Tensor({m}, 1.0f, backend));

		for (size_t i = 1; i < n; i++) {
			REQUIRE(result[i] == Tensor({m}, 0.0f, backend));
		}
	}
}

TEST_CASE("isClose views", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3, 10}, 1.0f, backend);
		Tensor b({3, 10}, 1.0f, backend);

		b[1][5] = 1.0f + 1e-6f;

		SECTION("View-View") {
			Tensor result = a[0].isClose(b[0]);
			REQUIRE(result == Tensor({10}, 1.0f, backend));
		}

		SECTION("Tensor-View") {
			Tensor result = a.isClose(b[0]);
			REQUIRE(result == Tensor({3, 10}, 1.0f, backend));
		}

		SECTION("View-Tensor") {
			Tensor result = a[0].isClose(b);
			REQUIRE(result == Tensor({3, 10}, 1.0f, backend));
		}
	}
}

TEST_CASE("isClose shape and errors", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		SECTION("Result shape") {
			Tensor a({3, 4}, 1.0f, backend);
			Tensor b({3, 4}, 1.0f, backend);

			Tensor result = a.isClose(b);
			REQUIRE(result.getShape() == Tensor::Shape({3, 4}));
			REQUIRE(result.getBackend() == backend);

			Tensor scalar(1.0f, backend);
			Tensor resultScalar = a.isClose(scalar);
			REQUIRE(resultScalar.getShape() == Tensor::Shape({3, 4}));
		}

		SECTION("Incompatible shapes") {
			Tensor a({10, 10}, 1.0f, backend);
			Tensor b({5, 5}, 1.0f, backend);

			REQUIRE_THROWS(a.isClose(b));
		}
	}
}