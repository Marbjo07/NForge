#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "nforge/core/tensor.h"

#define NFORGE_ENABLE_CUDA

static Backend getBackendFromIndex(int i) {
    switch (i) {
        case 0: return Backend::CPU;
#ifdef NFORGE_ENABLE_CUDA
        case 1: return Backend::CUDA;
#endif
        default: return Backend::CPU;
    }
}

static const char* backendName(Backend b) {
    switch (b) {
        case Backend::CPU:  return "CPU";
#ifdef NFORGE_ENABLE_CUDA
        case Backend::CUDA: return "CUDA";
#endif
        default: return "Unknown";
    }
}

TEST_CASE("Arithmetic ops behave identically on all backends", "[tensor][arith][backend]") {
    auto backendIndex = GENERATE(0
#ifdef NFORGE_ENABLE_CUDA
        , 1
#endif
    );

    Backend backend = getBackendFromIndex(backendIndex);

    DYNAMIC_SECTION("Backend = " << backendName(backend)) {
        Tensor a({5}, 4.0f, backend);
        Tensor b({5}, 1.5f, backend);

        Tensor add = a + b;
        Tensor sub = a - b;
        Tensor mul = a * b;

        for (size_t i = 0; i < 5; i++) {
            REQUIRE(add[i] == Tensor(5.5f));
            REQUIRE(sub[i] == Tensor(2.5f));
            REQUIRE(mul[i] == Tensor(6.0f));
        }
    }
}

TEST_CASE("Scalar broadcasting works on all backends", "[tensor][arith][broadcast]") {
    auto backendIndex = GENERATE(0
#ifdef NFORGE_ENABLE_CUDA
        , 1
#endif
    );

    Backend backend = getBackendFromIndex(backendIndex);

    DYNAMIC_SECTION("Backend = " << backendName(backend)) {
        Tensor a({4}, 2.0f, backend);
        Tensor s(3.0f);

        Tensor x = a + s;
        Tensor y = s + a;
        Tensor z = a * s;

        for (size_t i = 0; i < 4; i++) {
            REQUIRE(x[i] == Tensor(5.0f));
            REQUIRE(y[i] == Tensor(5.0f));
            REQUIRE(z[i] == Tensor(6.0f));
        }
    }
}

TEST_CASE("2D arithmetic consistency across backends", "[tensor][arith][2d]") {
    auto backendIndex = GENERATE(0
#ifdef NFORGE_ENABLE_CUDA
        , 1
#endif
    );

    Backend backend = getBackendFromIndex(backendIndex);

    auto rows = GENERATE(1ull, 3ull, 7ull);
    auto cols = GENERATE(1ull, 5ull, 11ull);

    DYNAMIC_SECTION(
        "Backend=" << backendName(backend)
        << " rows=" << rows
        << " cols=" << cols
    ) {
        Tensor A({rows, cols}, 2.0f, backend);
        Tensor B({rows, cols}, 0.5f, backend);

        Tensor C = A + B;
        Tensor D = A * B;

        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                REQUIRE(C[i][j] == Tensor(2.5f));
                REQUIRE(D[i][j] == Tensor(1.0f));
            }
        }
    }
}