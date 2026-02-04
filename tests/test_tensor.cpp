#include <catch2/catch_test_macros.hpp>
#include "nforge/tensor.h"

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