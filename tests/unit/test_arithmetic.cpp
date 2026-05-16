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

		for (size_t i = 0; i < 5; i++) {
			REQUIRE(add[i] == Tensor(5.5f, backend));
			REQUIRE(sub[i] == Tensor(2.5f, backend));
			REQUIRE(mul[i] == Tensor(6.0f, backend));
			REQUIRE(div[i] == Tensor(4.0f / 1.5f, backend));
		}
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

		for (size_t i = 0; i < 4; i++) {
			REQUIRE(x[i] == Tensor(5.0f, backend));
			REQUIRE(y[i] == Tensor(5.0f, backend));
			REQUIRE(z[i] == Tensor(6.0f, backend));
		}
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

		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				REQUIRE(C[i][j] == Tensor(2.5f, backend));
				REQUIRE(D[i][j] == Tensor(1.0f, backend));
			}
		}
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