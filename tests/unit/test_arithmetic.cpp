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

TEST_CASE("Matrix multiplication basic 2D", "[Tensor][Matmul]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		// A (2x3) * B (3x2) = C (2x2)
		Tensor A({2, 3}, 1.0f, backend);
		Tensor B({3, 2}, 2.0f, backend);

		Tensor C = A.matmul(B);

		// each element = 1*2 + 1*2 + 1*2 = 6
		REQUIRE(C == Tensor({2, 2}, 6.0f, backend));
	}
}

TEST_CASE("Matrix multiplication inner dimension mismatch throws", "[Tensor][Matmul]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor A({2, 3}, 1.0f, backend);
		Tensor B({4, 2}, 1.0f, backend);

		REQUIRE_THROWS(A.matmul(B));
	}
}

TEST_CASE("Matrix multiplication batched 3D", "[Tensor][Matmul]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		// A (2x2x3) * B (2x3x2) = C (2x2x2)
		Tensor A({2, 2, 3}, 1.0f, backend);
		Tensor B({2, 3, 2}, 2.0f, backend);

		Tensor C = A.matmul(B);

		// each element = 1*2 + 1*2 + 1*2 = 6
		REQUIRE(C == Tensor({2, 2, 2}, 6.0f, backend));
	}
}

TEST_CASE("Matrix multiplication batched vs non-batched", "[Tensor][Matmul]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor A({2, 2, 3}, 1.0f, backend);
		Tensor B({3, 2}, 2.0f, backend);

		Tensor C = A.matmul(B);

		REQUIRE(C == Tensor({2, 2, 2}, 6.0f, backend));
	}
}

TEST_CASE("Matrix multiplication mismatched batch throws", "[Tensor][Matmul]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor A({2, 2, 3}, 1.0f, backend);
		Tensor B({3, 3, 2}, 1.0f, backend);

		REQUIRE_THROWS(A.matmul(B));
	}
}

TEST_CASE("Matrix multiplication invalid rank throws", "[Tensor][Matmul]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor A({3}, 1.0f, backend);
		Tensor B({3, 2}, 1.0f, backend);

		REQUIRE_THROWS(A.matmul(B));
	}
}

TEST_CASE("Matrix multiplication non-uniform values", "[Tensor][Matmul]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor A({2, 2}, 0.0f, backend);
		A[0][0] = 1.0f;
		A[0][1] = 2.0f;
		A[1][0] = 3.0f;
		A[1][1] = 4.0f;

		Tensor B({2, 2}, 0.0f, backend);
		B[0][0] = 5.0f;
		B[0][1] = 6.0f;
		B[1][0] = 7.0f;
		B[1][1] = 8.0f;

		Tensor C = A.matmul(B);

		REQUIRE(C[0][0] == Tensor(19.0f, backend));
		REQUIRE(C[0][1] == Tensor(22.0f, backend));
		REQUIRE(C[1][0] == Tensor(43.0f, backend));
		REQUIRE(C[1][1] == Tensor(50.0f, backend));
	}
}

#ifdef NFORGE_ENABLE_CUDA
TEST_CASE("Matrix multiplcation equal across backends", "[Tensor][Matmul]") {
	size_t n = 5;
	size_t b = 7;
	Tensor orig({b, n, n}, Backend::CPU);
	orig.fillRand();


	Tensor cpu({b, n, n}, Backend::CPU);
	Tensor cuda({b, n, n}, Backend::CUDA);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			for (int k = 0; k < n; k++) {
				cpu[i][j][k] = orig[i][j][k].copy().toVector()[0];
				cuda[i][j][k] = orig[i][j][k].copy().toVector()[0];
			}
		}
	}

	for (int iter = 0; iter < 3; iter++) {
		cpu = cpu.matmul(cpu);
		cuda = cuda.matmul(cuda);
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			for (int k = 0; k < n; k++) {
				float lhs = cpu[i][j][k].copy().toVector()[0];
				float rhs = cuda[i][j][k].copy().toVector()[0];


				float dif = abs(lhs - rhs);
				CHECK(dif < 1e-5);
			}
		}
	}
}

#endif