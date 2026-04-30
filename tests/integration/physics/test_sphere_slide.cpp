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

    REQUIRE(res.position == create2dVector(0.734628736972809,  0.678469479084015));
    REQUIRE(res.speed    == create2dVector(1.203298449516296, -1.300454020500183));
    REQUIRE(res.t == 3.712913275f);
}