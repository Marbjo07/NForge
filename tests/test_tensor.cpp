#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "nforge/core/tensor.h"

TEST_CASE("Create tensor", "[tensor]") {
	Tensor t({3}, 4.0f, Backend::CPU);

	REQUIRE(t.numElements() == 3);
	REQUIRE(t.backendString() == "CPU");
}

TEST_CASE("Compare tensor", "[tensor]") {
	Tensor a({3, 9, 7}, 19.0f, Backend::CPU);
	Tensor b({3, 9, 7}, 19.0f, Backend::CPU);

	REQUIRE(a == b);
	REQUIRE(b == a);
}

TEST_CASE("Compare tensor views", "[tensor]") {
	Tensor a({3, 9, 7}, 19.0f, Backend::CPU);
	Tensor b({3, 9, 7}, 19.0f, Backend::CPU);

	auto x = a[0];
	auto y = b[0];

	REQUIRE(x == y);
	REQUIRE(y == x);
}

TEST_CASE("Compare tensor and tensor view", "[tensor]") {
	Tensor a({3, 9, 7}, 19.0f, Backend::CPU);
	Tensor b({9, 7}, 19.0f, Backend::CPU);

	auto x = a[0];
	auto y = a[1];

	REQUIRE(x == b);
	REQUIRE(x == y);
	
	REQUIRE(b == x);
	REQUIRE(b == y);

	REQUIRE(y == x);
	REQUIRE(y == b);
}

TEST_CASE("Add tensors", "[tensor]") {
	Tensor a({3}, 4.0f, Backend::CPU);
	Tensor b({3}, 1.0f, Backend::CPU);

	Tensor c = a + b;

	REQUIRE(c[0] == Tensor(5));
	REQUIRE(c[0] == c[1]);
	REQUIRE(c[1] == c[2]);
	REQUIRE(c[0] != Tensor(3));
}

TEST_CASE("Subtract and multiply tensors", "[tensor]") {
	Tensor a({3}, 4.0f, Backend::CPU);
	Tensor b({3}, 1.0f, Backend::CPU);

	Tensor sub = a - b;
	Tensor mul = a * b;

	REQUIRE(sub[0] == Tensor(3));
	REQUIRE(sub[0] == sub[1]);
	REQUIRE(sub[1] == sub[2]);

	REQUIRE(mul[0] == Tensor(4));
	REQUIRE(mul[0] == mul[1]);
	REQUIRE(mul[1] == mul[2]);
}

TEST_CASE("Broadcast scalar add", "[tensor]") {
	Tensor a({4}, 2.0f, Backend::CPU);
	Tensor s(3.0f);

	Tensor c = a + s;
	
	REQUIRE(c[2] == Tensor(5));
}

TEST_CASE("2D tensor shape and indexing", "[tensor]") {
	auto rows = GENERATE(1ull, 4ull, 10ull);
	auto cols = GENERATE(1ull, 10ull, 50ull);
	auto val = GENERATE(-1001.0f, 0.32f, 122.9f);

	DYNAMIC_SECTION("rows=" << rows << " cols=" << cols << " val=" << val) {
		Tensor a({rows, cols}, val, Backend::CPU);

		// Check number of elements
		REQUIRE(a.numElements() == rows * cols);


		// Compare slices
		REQUIRE(a[0] == Tensor({cols}, val, Backend::CPU));
		REQUIRE(a[0][0] == Tensor(val));
		REQUIRE(a[0][0] != Tensor(val - 1));

		REQUIRE_FALSE(a[0][0] != Tensor(val));
		REQUIRE_FALSE(a[0][0] == Tensor(val - 1));

		REQUIRE(a[0] == a[rows - 1]);
	}
}

TEST_CASE("Tensor slice assign", "[tensor]") {
	auto rows = GENERATE(1ull, 2ull, 3ull);
	auto cols = GENERATE(1ull, 4ull, 8ull);
	auto val = GENERATE(0.0f, 1.5f);

	DYNAMIC_SECTION("rows=" << rows << " cols=" << cols << " val=" << val) {
		
		// Create A and random B
		Tensor A({rows, cols}, val, Backend::CPU);
		Tensor B({rows, cols}, Backend::CPU);
		B.fillRand();
		
		// Slice copy
		for (size_t i = 0; i < rows; i++) {
			A[i] = B[i];
		}

		// Element wise compare
		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				REQUIRE(A[i][j] == B[i][j]);
			}
		}
		
		INFO("A=" + A.dataString() + "\nB=" + B.dataString());
		REQUIRE(A == B);
	}
}