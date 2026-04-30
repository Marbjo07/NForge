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

TEST_CASE("Chained view assignment", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor A({3, 4}, 1.0f, backend);
        Tensor B({3, 4}, 2.0f, backend);
        Tensor C({3, 4}, 3.0f, backend);

        // A[0] = B[0] = C[0] should make both rows equal to C[0]
        A[0] = B[0] = C[0];

        REQUIRE(B[0] == C[0]);
        REQUIRE(A[0] == C[0]); 
        REQUIRE(A[0] == B[0]);
    }
}