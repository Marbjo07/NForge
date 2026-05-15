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
                a[i][j] = Tensor(i * 3 + j, backend);
            }
        }
        // a = [[0, 1, 2], [3, 4, 5]]
        float meanBlock1 = (0 + 1 + 2) / 3.0f;
        float meanBlock2 = (3 + 4 + 5) / 3.0f;
        float mean = (meanBlock1 + meanBlock2) / 2.0f;

        Tensor reduced({2}, 0, backend);
        reduced[0] = Tensor(meanBlock1, backend);
        reduced[1] = Tensor(meanBlock2, backend);

        REQUIRE(a.mean() == Tensor(mean, backend));
        REQUIRE(a.mean(0) == Tensor(mean, backend)); // dim=0 is default
        REQUIRE(a.mean(1) == reduced);
        REQUIRE(a.mean(2) == a); // mean of each element is it self
    }
}

TEST_CASE("Tensor sum reduction", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({2, 3}, backend);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                a[i][j] = Tensor(i * 3 + j, backend);
            }
        }
        // a = [[0, 1, 2], [3, 4, 5]]
        float sumBlock1 = 0 + 1 + 2;
        float sumBlock2 = 3 + 4 + 5;
        float sum = sumBlock1 + sumBlock2;

        Tensor reduced({2}, 0, backend);
        reduced[0] = Tensor(sumBlock1, backend);
        reduced[1] = Tensor(sumBlock2, backend);

        REQUIRE(a.sum() == Tensor(sum, backend));
        REQUIRE(a.sum(0) == Tensor(sum, backend)); // dim=0 is default
        REQUIRE(a.sum(1) == reduced);
        REQUIRE(a.sum(2) == a); // sum of each element is it self
    }
}

TEST_CASE("Tensor max reduction", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({2, 3}, backend);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                a[i][j] = Tensor(i * 3 + j, backend);
            }
        }
        // a = [[0, 1, 2], [3, 4, 5]]
        float maxBlock1 = 2;
        float maxBlock2 = 5;
        float max = std::max(maxBlock1, maxBlock2);

        Tensor reduced({2}, 0, backend);
        reduced[0] = maxBlock1;
        reduced[1] = maxBlock2;

        REQUIRE(a.max() == Tensor(max, backend));
        REQUIRE(a.max(0) == Tensor(max, backend)); // dim=0 is default
        REQUIRE(a.max(1) == reduced);
        REQUIRE(a.max(2) == a); // max of each element is it self
    }
}

TEST_CASE("Tensor min reduction", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({2, 3}, backend);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                a[i][j] = Tensor(i * 3 + j, backend);
            }
        }
        // a = [[0, 1, 2], [3, 4, 5]]
        float minBlock1 = 0;
        float minBlock2 = 3;
        float min = std::min(minBlock1, minBlock2);

        Tensor reduced({2}, 0, backend);
        reduced[0] = minBlock1;
        reduced[1] = minBlock2;

        REQUIRE(a.min() == Tensor(min, backend));
        REQUIRE(a.min(0) == Tensor(min, backend)); // dim=0 is default
        REQUIRE(a.min(1) == reduced);
        REQUIRE(a.min(2) == a); // min of each element is it self
    }
}

TEST_CASE("Tensor prod reduction", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({2, 3}, backend);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                a[i][j] = Tensor(i * 3 + j, backend);
            }
        }
        // a = [[0, 1, 2], [3, 4, 5]]
        float prodBlock1 = 0 * 1 * 2;
        float prodBlock2 = 3 * 4 * 5;
        float prod = prodBlock1 * prodBlock2;

        Tensor reduced({2}, 0, backend);
        reduced[0] = prodBlock1;
        reduced[1] = prodBlock2;

        REQUIRE(a.prod() == Tensor(prod, backend));
        REQUIRE(a.prod(0) == Tensor(prod, backend)); // dim=0 is default
        REQUIRE(a.prod(1) == reduced);
        REQUIRE(a.prod(2) == a); // prod of each element is it self
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
            Tensor sum = a.sum(d);

            REQUIRE(sum != Tensor(0, backend)); // sanity check

            // TODO: refactor with relative comparison and tolerance
            Tensor targetMean = sum / count;
            Tensor diff = a.mean(d) - targetMean;
            Tensor absDiff = diff * diff;
            Tensor maxDiff = absDiff.max();
            REQUIRE(maxDiff.toVector()[0] < 1e-6f);
            
            if (d != 3) {
                count /= shape[d];
            }
        }
    }
}

TEST_CASE("Tensor reduction with stride", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        size_t n = 4;
        size_t m = 6;
        Tensor a({n, m}, backend);
        
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < m; j++) {
                a[i][j] = Tensor(i * m + j, backend);
            }
        }
        // a = [[ 0,  1,  2,  3,  4,  5], 
        //      [ 6,  7,  8,  9, 10, 11],
        //      [12, 13, 14, 15, 16, 17],
        //      [18, 19, 20, 21, 22, 23]]
        
        auto b = a.subsample({2, 2}).copy();
        
        // b = [[ 0,  2,  4],
        //      [12, 14, 16]]

        REQUIRE(b.mean(0) == Tensor(8.0f, backend));
        REQUIRE(b.mean(1)[0] == Tensor(2.0f, backend));
        REQUIRE(b.mean(1)[1] == Tensor(14.0f, backend));
        
        REQUIRE(b.sum(0) == Tensor(48.0f, backend));
        REQUIRE(b.sum(1)[0] == Tensor(6.0f, backend));
        REQUIRE(b.sum(1)[1] == Tensor(42.0f, backend));
    }
}