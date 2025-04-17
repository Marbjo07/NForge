#include <catch2/catch_test_macros.hpp>
#include "nforge/tensor.h"

using namespace nforge;

TEST_CASE("Tensor basic fill and get", "[tensor]") {
	Tensor t(3);
	t.fill(1.5f);
	REQUIRE(t.get(0) == 1.5f);
	REQUIRE(t.get(2) == 1.5f);
}

TEST_CASE("Tensor out of bounds", "[tensor]") {
	Tensor t(3);
	REQUIRE_THROWS_AS(t.get(10), std::out_of_range);
}	
