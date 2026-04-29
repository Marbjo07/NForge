#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "nforge/nforge.h"
#include "utils.h"


TEST_CASE("Tensor mean reduction", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({2, 3}, backend);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                a[i][j] = i * 3 + j;
            }
        }
        // a = [[0, 1, 2], [3, 4, 5]]
        float meanBlock1 = (0 + 1 + 2) / 3.0f;
        float meanBlock2 = (3 + 4 + 5) / 3.0f;
        float mean = (meanBlock1 + meanBlock2) / 2.0f;

        Tensor reduced({2}, backend);
        reduced[0] = meanBlock1;
        reduced[1] = meanBlock2;

        REQUIRE(a.mean() == Tensor(mean, backend));
        REQUIRE(a.mean(dim=0) == Tensor(mean, backend)); // dim=0 is default
        REQUIRE(a.mean(dim=1) == reduced);
        REQUIRE(a.mean(dim=2) == a); // mean of each element is it self
    }
}

TEST_CASE("Tensor sum reduction", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({2, 3}, backend);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                a[i][j] = i * 3 + j;
            }
        }
        // a = [[0, 1, 2], [3, 4, 5]]
        float sumBlock1 = 0 + 1 + 2;
        float sumBlock2 = 3 + 4 + 5;
        float sum = sumBlock1 + sumBlock2;

        Tensor reduced({2}, backend);
        reduced[0] = sumBlock1;
        reduced[1] = sumBlock2;

        REQUIRE(a.sum() == Tensor(sum, backend));
        REQUIRE(a.sum(dim=0) == Tensor(sum, backend)); // dim=0 is default
        REQUIRE(a.sum(dim=1) == reduced);
        REQUIRE(a.sum(dim=2) == a); // sum of each element is it self
    }
}

TEST_CASE("Tensor max reduction", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({2, 3}, backend);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                a[i][j] = i * 3 + j;
            }
        }
        // a = [[0, 1, 2], [3, 4, 5]]
        float maxBlock1 = 2;
        float maxBlock2 = 5;
        float max = max(maxBlock1, maxBlock2);

        Tensor reduced({2}, backend);
        reduced[0] = maxBlock1;
        reduced[1] = maxBlock2;

        REQUIRE(a.max()) == Tensor(max, backend));
        REQUIRE(a.max(dim=0) == Tensor(max, backend)); // dim=0 is default
        REQUIRE(a.max(dim=1) == reduced);
        REQUIRE(a.max(dim=2) == a); // max of each element is it self
    }
}

TEST_CASE("Tensor min reduction", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({2, 3}, backend);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                a[i][j] = i * 3 + j;
            }
        }
        // a = [[0, 1, 2], [3, 4, 5]]
        float minBlock1 = 0;
        float minBlock2 = 3;
        float min = std::min(minBlock1, minBlock2);

        Tensor reduced({2}, backend);
        reduced[0] = minBlock1;
        reduced[1] = minBlock2;

        REQUIRE(a.min() == Tensor(min, backend));
        REQUIRE(a.min(dim=0) == Tensor(min, backend)); // dim=0 is default
        REQUIRE(a.min(dim=1) == reduced);
        REQUIRE(a.min(dim=2) == a); // min of each element is it self
    }
}

TEST_CASE("Tensor prod reduction", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({2, 3}, backend);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                a[i][j] = i * 3 + j;
            }
        }
        // a = [[0, 1, 2], [3, 4, 5]]
        float prodBlock1 = 0 * 1 * 2;
        float prodBlock2 = 3 * 4 * 5;
        float min = prodBlock1 * prodBlock2;

        Tensor reduced({2}, backend);
        reduced[0] = prodBlock1;
        reduced[1] = prodBlock2;

        REQUIRE(a.prod() == Tensor(prod, backend));
        REQUIRE(a.prod(dim=0) == Tensor(prod, backend)); // dim=0 is default
        REQUIRE(a.prod(dim=1) == reduced);
        REQUIRE(a.prod(dim=2) == a); // prod of each element is it self
    }
}

TEST_CASE("Tensor mean reduction by sum reduction", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        std::vector<size_t> shape = {4, 5, 8};
        
        Tensor a(shape, backend);
        a.fillRand();
            
        size_t count = 4 * 5 * 8;
        for (size_t d = 0; d <= 3; d++) {
            float sum = a.sum(dim=d);
            REQUIRE(a.mean(dim=d) == sum / count);
            
            if (d != 3) {
                count /= shape[d];
            }
        }
    }
}
