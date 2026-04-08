#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "nforge/nforge.h"
#include "utils.h"


TEST_CASE("View shape", "[View]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({3, 6, 6, 6}, 1.0f, backend);
        Tensor::View b(a, {1, 3, 2});

        REQUIRE(b.getShape() == Tensor::Shape({6}));
    }
}
