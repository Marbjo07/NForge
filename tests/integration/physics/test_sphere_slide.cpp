#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "physics/sphere_slide.h"
#include "utils.h"

TEST_CASE("Cube slide of sphere", "[Physics]") {
	SphereSlideResults res = simulateSphereSlide(SphereSlideParams{});

	INFO(res.position.getDataString());
	INFO(res.speed.getDataString());
	INFO(res.t);

	REQUIRE(res.position.isClose(create2dVector(0.73462, 0.67846)).toVector()[0]);
	REQUIRE(res.speed.isClose(create2dVector(1.20329, -1.30045)).toVector()[0]);

	REQUIRE(res.t == 3.712913275f);
}