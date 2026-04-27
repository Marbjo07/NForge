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
    float detachX =  std::sqrt(5.0f) / 3.0f;
    float detachY =  2.0f / 3.0f;
    float detachSpeed =  std::sqrt(2.0f * 9.81f / 3.0f);
    float detachSpeedX  =  detachSpeed * (2.0f / 3.0f);
    float detachSpeedY  = -detachSpeed * std::sqrt(5.0f) / 3.0f;


    SphereSlideResults res = simulateSphereSlide(SphereSlideParams{});
    
    REQUIRE(res.approxEqual({
        create2dVector(detachX, detachY),
        create2dVector(detachSpeedX, detachSpeedY),
        3.71291f
    }));
}