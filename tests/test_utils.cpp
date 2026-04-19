#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "nforge/nforge.h"
#include "utils.h"


#ifdef NFORGE_WITH_CUDA
bool cudaEnabled = true;

#else // without cuda
bool cudaEnabled = false;

#endif


TEST_CASE("Not cuda when disabled", "[Utils]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        if (!cudaEnabled) {
            REQUIRE(backend != Backend::CUDA);
        }
    }
}
