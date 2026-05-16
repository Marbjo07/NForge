#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cmath>

#include "nforge/nforge.h"
#include "utils.h"

TEST_CASE("Create tensor", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor t({3}, 4.0f, backend);

		REQUIRE(t.getNumElements() == 3);
	}
}

TEST_CASE("Compare tensor", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3, 9, 7}, 19.0f, backend);
		Tensor b({3, 9, 7}, 19.0f, backend);
		Tensor c({3, 7, 9}, 19.0f, backend);

		REQUIRE(a == b);
		REQUIRE(b == a);
		REQUIRE(c != a);
		REQUIRE(c != b);
	}
}

TEST_CASE("Compare tensor views", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3, 9, 7}, 19.0f, backend);
		Tensor b({3, 9, 7}, 19.0f, backend);

		auto x = a[0];
		auto y = b[0];

		REQUIRE(x == y);
		REQUIRE(y == x);
	}
}

TEST_CASE("Compare tensor and tensor view", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3, 9, 7}, 19.0f, backend);
		Tensor b({9, 7}, 0.0f, backend);

		auto x = a[0];
		auto y = a[1];

		REQUIRE(x != b);
		REQUIRE(x == y);

		REQUIRE(b != x);
		REQUIRE(b != y);

		REQUIRE(y == x);
		REQUIRE(y != b);
	}
}

TEST_CASE("Tensor view copy and compare", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3, 9, 7}, 19.0f, backend);
		Tensor b({9, 7}, 0.0f, backend);
		Tensor c({3, 9, 7}, 19.0f, backend);

		a[0] = b;
		a[1] = b;

		REQUIRE(a != c);
		REQUIRE(a[0] != c[0]);
		REQUIRE(a[2] == c[1]);

		REQUIRE(c != a);
		REQUIRE(c[0] != a[0]);
		REQUIRE(c[0] == a[2]);

		REQUIRE(a[1] == b);
		REQUIRE(a[2] != b);

		REQUIRE(b == a[1]);
		REQUIRE(b != a[2]);
	}
}

TEST_CASE("Add tensors", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3}, 4.0f, backend);
		Tensor b({3}, 1.0f, backend);

		Tensor c = a + b;

		REQUIRE(c[0] == Tensor(5.0f, backend));
		REQUIRE(c[0] == c[1]);
		REQUIRE(c[1] == c[2]);
		REQUIRE(c[0] != Tensor(3.0f, backend));
	}
}

TEST_CASE("Subtract and multiply tensors", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3}, 4.0f, backend);
		Tensor b({3}, 1.0f, backend);

		Tensor sub = a - b;
		Tensor mul = a * b;

		REQUIRE(sub[0] == Tensor(3.0f, backend));
		REQUIRE(sub[0] == sub[1]);
		REQUIRE(sub[1] == sub[2]);

		REQUIRE(mul[0] == Tensor(4.0f, backend));
		REQUIRE(mul[0] == mul[1]);
		REQUIRE(mul[1] == mul[2]);
	}
}

TEST_CASE("Broadcast scalar add", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({4}, 2.0f, backend);
		Tensor s(3.0f, backend);

		Tensor x = a + s;
		Tensor y = s + a;

		for (size_t i = 0; i < 4; i++) {
			REQUIRE(x[i] == Tensor(5.0f, backend));
			REQUIRE(y[i] == Tensor(5.0f, backend));
		}
	}
}

TEST_CASE("2D tensor shape and indexing", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		auto rows = GENERATE(1ul, 4ul, 10ul);
		auto cols = GENERATE(1ul, 10ul, 50ul);
		auto val = GENERATE(-1001.0f, 0.32f, 122.9f);

		DYNAMIC_SECTION("rows=" << rows << " cols=" << cols << " val=" << val) {
			Tensor a({rows, cols}, val, backend);

			// Check number of elements
			REQUIRE(a.getNumElements() == rows * cols);

			// Compare views
			REQUIRE(a[0] == Tensor({cols}, val, backend));
			REQUIRE(a[0][0] == Tensor(val, backend));
			REQUIRE(a[0][0] != Tensor(val - 1, backend));

			REQUIRE_FALSE(a[0][0] != Tensor(val, backend));
			REQUIRE_FALSE(a[0][0] == Tensor(val - 1, backend));

			REQUIRE(a[0] == a[rows - 1]);
		}
	}
}

TEST_CASE("Chained tensor assignment", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3}, 1.0f, backend);
		Tensor b({3}, 2.0f, backend);
		Tensor c({3}, 3.0f, backend);

		// a = b = c should make both a and b equal to c
		a = b = c;

		REQUIRE(b == c);
		REQUIRE(a == c);
		REQUIRE(a == b);
	}
}

TEST_CASE("Tensor view assign", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		auto rows = GENERATE(1ul, 2ul, 3ul);
		auto cols = GENERATE(1ul, 4ul, 8ul);
		auto val = GENERATE(0.0f, 1.5f);

		DYNAMIC_SECTION("rows=" << rows << " cols=" << cols << " val=" << val) {
			// Create A and random B
			Tensor A({rows, cols}, val, backend);
			Tensor B({rows, cols}, backend);
			B.fillRand();

			// View copy
			for (size_t i = 0; i < rows; i++) {
				A[i] = B[i];
			}

			// Element wise compare
			for (size_t i = 0; i < rows; i++) {
				for (size_t j = 0; j < cols; j++) {
					REQUIRE(A[i][j] == B[i][j]);
				}
			}

			REQUIRE(A == B);
		}
	}
}

TEST_CASE("Tensor and float add", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({4}, 2.0f, backend);
		Tensor b = a + 3;

		REQUIRE(b == Tensor({4}, 5.0f, backend));
	}
}

TEST_CASE("Tensor and float sub", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({4}, 2.0f, backend);
		Tensor b = a - 3;

		REQUIRE(b == Tensor({4}, -1.0f, backend));
	}
}

TEST_CASE("Tensor and float mul", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({4}, 2.0f, backend);
		Tensor b = a * 3;

		REQUIRE(b == Tensor({4}, 6.0f, backend));
	}
}

TEST_CASE("Tensor and float div", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({4}, 2.0f, backend);
		Tensor b = a / 3;

		REQUIRE(b == Tensor({4}, 2.0f / 3.0f, backend));
	}
}

TEST_CASE("Float and Tensor add", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({4}, 2.0f, backend);
		Tensor b = 3 + a;

		REQUIRE(b == Tensor({4}, 5.0f, backend));
	}
}

TEST_CASE("Float and Tensor sub", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({4}, 2.0f, backend);
		Tensor b = 3 - a;

		REQUIRE(b == Tensor({4}, 1.0f, backend));
	}
}

TEST_CASE("Float and Tensor mul", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({4}, 2.0f, backend);
		Tensor b = 3 * a;

		REQUIRE(b == Tensor({4}, 6.0f, backend));
	}
}

TEST_CASE("Float and Tensor div", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({4}, 2.0f, backend);
		Tensor b = 3 / a;

		REQUIRE(b == Tensor({4}, 1.5f, backend));
	}
}

TEST_CASE("Invalid Tensor index throws", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({2}, 2.0f, backend);
		REQUIRE_THROWS(a[3]);
	}
}

TEST_CASE("Invalid Tensor 2D index throws", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({2, 6}, 2.0f, backend);

		REQUIRE_THROWS(a[0][6]);
		REQUIRE_THROWS(a[2][4]);
		REQUIRE_THROWS(a[1][6]);
		REQUIRE_THROWS(a[-1][0]);
		REQUIRE_THROWS(a[-1][-1]);
	}
}

TEST_CASE("Scalar Tensor initialization", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		for (size_t n : {1, 5, 8, 16}) {
			Tensor a({n}, backend);
			REQUIRE(a.getShape() == Tensor::Shape({n}));

			Tensor b({n}, 1.0f, backend);
			REQUIRE(a.getShape() == Tensor::Shape({n}));
		}

		Tensor a(1.0f, backend);
		REQUIRE(a.getShape() == Tensor::Shape({1}));
	}
}

TEST_CASE("Scalar assignment with float", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a(1.0f, backend);
		a = 3.0f;

		REQUIRE(a == Tensor(3.0f, backend));
	}
}

TEST_CASE("Indexed scalar assignment with float", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3, 4}, 0.0f, backend);
		a[2][3] = 3.0f;

		REQUIRE(a[2][3] == Tensor(3.0f, backend));
		REQUIRE(a != Tensor({3, 4}, 0.0f, backend));
		REQUIRE(a[0] == Tensor({4}, 0.0f, backend));
		REQUIRE(a[1] == Tensor({4}, 0.0f, backend));
	}
}

TEST_CASE("Verify frobenius norm return tensor", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({4, 5}, backend);

		Tensor norm = a.norm();

		REQUIRE(norm.getShape() == Tensor::Shape({1}));
		REQUIRE(norm.getBackend() == backend);
	}
}

TEST_CASE("Frobenius norm of scalar", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		SECTION("Positive scalar") {
			Tensor a(4.0f, backend);
			REQUIRE(a.norm() == Tensor(4.0f, backend));
		}

		SECTION("Negative scalar") {
			Tensor b(-4.0f, backend);
			REQUIRE(b.norm() == Tensor(4.0f, backend));
		}
	}
}

TEST_CASE("Frobenius norm of vector", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		SECTION("Uniform values") {
			Tensor a({5}, -4.0f, backend);

			// TODO: refactor with relative comparison and tolerance
			Tensor diff = a.norm() - Tensor(4.0f * std::sqrt(5.0f), backend);
			REQUIRE(abs(diff.toVector()[0]) < 1e-6f);
		}

		SECTION("Sequential values [0, 1, 2, 3, 4]") {
			Tensor a({5}, 0.0f, backend);
			for (size_t i = 0; i < 5; i++) a[i] = i;

			// TODO: refactor with relative comparison and tolerance
			Tensor diff = a.norm() - Tensor(std::sqrt(30.0f), backend);
			REQUIRE(abs(diff.toVector()[0]) < 1e-6f);
		}
	}
}

TEST_CASE("Frobenius norm of matrix", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		SECTION("Uniform values") {
			Tensor a({5, 8}, -4.0f, backend);
			REQUIRE(a.norm() == Tensor(4.0f * std::sqrt(5.0f * 8.0f), backend));
		}

		SECTION("Sequential values") {
			Tensor a({5, 8}, 0.0f, backend);

			float sum = 0;
			for (size_t i = 0; i < 5; i++) {
				for (size_t j = 0; j < 8; j++) {
					a[i][j] = i * 8 + j;
					sum += std::pow(i * 8 + j, 2);
				}
			}

			REQUIRE(a.norm() == Tensor(std::sqrt(sum), backend));
		}
	}
}

TEST_CASE("Frobenius norm of 4-rank tensor", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		SECTION("Uniform values") {
			Tensor a({8, 9, 4, 3}, -4.0f, backend);
			REQUIRE(a.norm() == Tensor(4.0f * std::sqrt(8.0f * 9.0f * 4.0f * 3.0f), backend));
		}

		SECTION("Random values") {
			Tensor a({8, 9, 4, 3}, backend);

			a.fillRand();

			float sum = 0;
			const auto elements = a.toVector();
			for (float e : elements) {
				sum += e * e;
			}

			// random tensor has a lot of noise, so use relative error
			Tensor ratio = a.norm() / std::sqrt(sum);
			REQUIRE(ratio.toVector()[0] > 0.99);
			REQUIRE(ratio.toVector()[0] < 1.01);
		}
	}
}

TEST_CASE("Assign Tensor::View to Tensor", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3}, 0.0f, backend);
		Tensor b({5, 4}, 1.0f, backend);

		// Basic assignment from view
		a = b[0];

		// Correct shape changes
		REQUIRE(a.getShape() == Tensor::Shape({4}));
		// Correct data after assignment
		REQUIRE(a == Tensor({4}, 1.0f, backend));
		// Verify if it acts as a copy
		b[0][0] = 99.0f;
		REQUIRE(a[0] == Tensor(1.0f, backend));
	}
}