#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iomanip>

#include "physics/projectile_motion.h"
#include "utils.h"

TEST_CASE("Diagonal throw", "[Physics]") {
	ProjectileMotionParams params;
	params.angle = 45;  // degrees

	std::cout << std::fixed << std::setprecision(15) << "\n";

	ProjectileMotionResults res = simulateProjectileMotion(params);

	INFO(res.position.getDataString());
	INFO(res.speed.getDataString());
	INFO(res.t);

	REQUIRE(res.position.isClose(create2dVector(10.1894, 0)).toVector()[0]);
	REQUIRE(res.speed.isClose(create2dVector(7.0710, -7.0651)).toVector()[0]);
	REQUIRE(res.t == 1.44101131f);
}

TEST_CASE("Vertical throw", "[Physics]") {
	ProjectileMotionParams params;
	params.angle = 90;  // degrees

	ProjectileMotionResults res = simulateProjectileMotion(params);

	INFO("position: " << res.position.getDataString());
	INFO("speed: " << res.speed.getDataString());
	INFO("t: " << res.t);

	REQUIRE(res.position.isClose(create2dVector(0, 0)).toVector()[0]);
	REQUIRE(res.speed.isClose(create2dVector(0, 10)).toVector()[0]);
	REQUIRE(res.t == 2.038034677505493);
}