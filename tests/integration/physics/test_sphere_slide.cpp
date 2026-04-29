#include "physics/sphere_slide.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>

Tensor create2dVector(float x, float y) {
    Tensor a({2}, 0.0f);
    a[0] = x;
    a[1] = y;
    return a;
}   

TEST_CASE("Cube slide of sphere", "[Physics]") {
    SphereSlideResults res = simulateSphereSlide(SphereSlideParams{});

    REQUIRE(res.position == create2dVector(0.73462,  0.67846));
    REQUIRE(res.speed    == create2dVector(1.20329, -1.30045));
    REQUIRE(res.t == 3.7129f);
}