#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "physics/sphere_slide.h"
#include "utils.h"

TEST_CASE("Cube slide of sphere", "[Physics]") {
	SphereSlideResults res = simulateSphereSlide(SphereSlideParams{});

	INFO(res.position.getDataString());
	INFO(res.speed.getDataString());
	INFO(res.t);

	REQUIRE(isSimilar(res.position, create2dVector(0.734628736972809, 0.678469479084015)));
	REQUIRE(isSimilar(res.speed, create2dVector(1.203298449516296, -1.300454020500183)));
	REQUIRE(res.t == 3.712913275f);
}