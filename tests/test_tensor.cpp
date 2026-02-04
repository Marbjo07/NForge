#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "nforge/core/tensor.h"

TEST_CASE("Create tensor", "[tensor]") {
	Tensor t({3}, 4.0f, Backend::CPU);

	REQUIRE(t.numElements() == 3);
	REQUIRE(t.backendString() == "CPU");
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

	REQUIRE(c[0] == Tensor(5));
	REQUIRE(c[3] == Tensor(5));
}

TEST_CASE("2D tensor shape and indexing", "[tensor]") {
    auto n = GENERATE(1, 4, 10);
    auto m = GENERATE(1, 10, 50);
	auto x = GENERATE(-1000, 0.32, 122.9);

    DYNAMIC_SECTION("n=" << n << " m=" << m << " val=" << x) {
		Tensor a({(size_t)n, (size_t)m}, x, Backend::CPU);

    	REQUIRE(a.numElements() == n * m);
		REQUIRE(a[0] == Tensor({(size_t)m}, x, Backend::CPU));
    	REQUIRE(a[0][0] == Tensor(x));
    	REQUIRE(a[0][0] != Tensor(x - 1));   
    	
		REQUIRE_FALSE(a[0][0] != Tensor(x));
		REQUIRE_FALSE(a[0][0] == Tensor(x - 1));

    	REQUIRE(a[0] == a[n - 1]);
	}
}