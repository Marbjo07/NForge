#include "physics/projectile_motion.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iomanip>

#include "utils.h"



TEST_CASE("Diagonal throw", "[Physics]") {
    ProjectileMotionParams params;
    params.angle = 45; // degrees

    std::cout << std::fixed << std::setprecision(15) << "\n";

    ProjectileMotionResults res = simulateProjectileMotion(params);

    REQUIRE(res.position == create2dVector(10.189493179321289, -0.002771137747914));
    REQUIRE(res.speed == create2dVector(7.071067810058594, -7.065103530883789));
    REQUIRE(res.t == 1.44101131f);
}


TEST_CASE("Vertical throw", "[Physics]") {
    ProjectileMotionParams params;
    params.angle = 90; // degrees

    ProjectileMotionResults res = simulateProjectileMotion(params);

    REQUIRE(res.position == create2dVector(-0.000000890855176, -0.002778571099043));
    REQUIRE(res.speed == create2dVector(-0.000000437113897, -9.992918014526367));
    REQUIRE(res.t == 2.038034677505493);
}