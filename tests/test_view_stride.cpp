#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "nforge/nforge.h"
#include "utils.h"


TEST_CASE("View shape with stride", "[View][Stride]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({3, 6, 6, 6}, 1.0f, backend);
        Tensor::View b = a.subsample({1, 2, 3, 6});

        REQUIRE(b.getShape() == Tensor::Shape({3, 3, 2, 1}));
        REQUIRE(b.getStride() == std::vector<size_t>({1, 2, 3, 6}));
    }
}


TEST_CASE("Extend stride to dimensions", "[View][Stride]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({3, 6, 6, 6}, 1.0f, backend);
        Tensor::View b = a.subsample({1, 3, 6, 1});

        REQUIRE(b.getShape() == Tensor::Shape({3, 2, 1, 6}));
    }
}

TEST_CASE("Throw on stride and shape mismatch", "[View][Stride]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({1, 2, 3, 4}, 1.0f, backend);
        
        REQUIRE_THROWS(Tensor::View(a, {}, {1, 2, 3}));
    }
}


TEST_CASE("Assign strided view", "[View][Stride]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({3, 6}, 1.0f, backend);
        Tensor::View b = a.subsample({1, 2});

        b = Tensor({3, 3}, 2.0f, backend);


        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 6; j++) {
                if (j % 2 == 0) {
                    REQUIRE(a[i][j] == Tensor(2.0f, backend));
                }
                else {
                    REQUIRE(a[i][j] == Tensor(1.0f, backend));
                }
            }
        }
    }
}

TEST_CASE("Zero strided view", "[View][Stride]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({5, 6}, 1.0f, backend);
        Tensor::View b = a.subsample({0});
        Tensor::View c = a[3].subsample({0});

        // should only assign the first element in a, a[0][0]
        b = Tensor(2.0f, backend);

        REQUIRE(a[0][0] == Tensor(2.0f, backend));
        REQUIRE(a[0][1] == Tensor(1.0f, backend));
        REQUIRE(a[1][1] == Tensor(1.0f, backend));
        REQUIRE(a[1][0] == Tensor(1.0f, backend));

        
        // should only assign the first element in a[3], a[3][0]
        c = Tensor(3.0f, backend);
        
        REQUIRE(a[3][0] == Tensor(2.0f, backend));
        REQUIRE(a[3][1] == Tensor(1.0f, backend));
        REQUIRE(a[3][2] == Tensor(1.0f, backend));
    }
}


// ---------------------------------------------------------------------------
// Copy with stride
// ---------------------------------------------------------------------------

TEST_CASE("Copy strided view produces dense tensor", "[View][Stride]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({4, 8}, 3.0f, backend);
        Tensor::View b = a.subsample({2, 4});

        Tensor c = b.copy();

        // shape should be the strided shape
        REQUIRE(c.getShape() == Tensor::Shape({4 / 2, 8 / 4}));

        REQUIRE(c == Tensor({2, 2}, 3.0f, backend));
    }
}

TEST_CASE("Copy of strided view is independent from parent", "[View][Stride]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({6}, 5.0f, backend);
        Tensor::View b = a.subsample({2});

        Tensor c = b.copy();
        a = Tensor({6}, 99.0f, backend);

        for (size_t i = 0; i < 3; i++) {
            REQUIRE(c[i] == Tensor(5.0f, backend));
        }
    }
}

// ---------------------------------------------------------------------------
// Strided view indexing
// ---------------------------------------------------------------------------

TEST_CASE("Index into strided view", "[View][Stride]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({4, 6}, 1.0f, backend);
        Tensor::View b = a.subsample({2, 3});

        // b has shape {2, 2}, indexing row 1 should give a view of length 2
        Tensor::View row = b[1];

        REQUIRE(row.getShape() == Tensor::Shape({2}));
    }
}

TEST_CASE("Position preserved through strided view index", "[View][Stride]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({6, 4}, 0.0f, backend);
        Tensor::View b = a.subsample({3, 2});

        // b shape is {2, 2}, b[1] should reference row 3 of the parent
        Tensor::View row = b[1];
        row = Tensor({2}, 7.0f, backend);

        REQUIRE(a[3][0] == Tensor(7.0f, backend));
        REQUIRE(a[3][2] == Tensor(7.0f, backend));

        // untouched elements
        REQUIRE(a[3][1] == Tensor(0.0f, backend));
        REQUIRE(a[3][3] == Tensor(0.0f, backend));
        REQUIRE(a[0][0] == Tensor(0.0f, backend));
    }
}

// ---------------------------------------------------------------------------
// Stride with position (combined index + stride)
// ---------------------------------------------------------------------------

TEST_CASE("Strided view with position offset", "[View][Stride]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({4, 8}, 1.0f, backend);
        Tensor::View b = a[1].subsample({1, 4});

        // positioned at row 1, stride {1, 4} on remaining {8} => shape {8/4} = {2}
        REQUIRE(b.getShape() == Tensor::Shape({2}));

        b = Tensor({2}, 9.0f, backend);

        REQUIRE(a[1][0] == Tensor(9.0f, backend));
        REQUIRE(a[1][4] == Tensor(9.0f, backend));
        REQUIRE(a[1][1] == Tensor(1.0f, backend));
        REQUIRE(a[0][0] == Tensor(1.0f, backend));
    }
}

TEST_CASE("Deeper position with stride", "[View][Stride]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({3, 6, 4}, 0.0f, backend);
        Tensor::View b = a[2][1].subsample({2});

        // position {2,1} selects a[2][1] which has shape {4}, stride {2} -> shape {2}
        REQUIRE(b.getShape() == Tensor::Shape({2}));

        b = Tensor({2}, 5.0f, backend);

        REQUIRE(a[2][1][0] == Tensor(5.0f, backend));
        REQUIRE(a[2][1][2] == Tensor(5.0f, backend));
        REQUIRE(a[2][1][1] == Tensor(0.0f, backend));
        REQUIRE(a[2][1][3] == Tensor(0.0f, backend));
    }
}

// ---------------------------------------------------------------------------
// Broadcast factory
// ---------------------------------------------------------------------------

TEST_CASE("Broadcast factory creates zero-strided view", "[View][Stride]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor scalar(4.0f, backend);
        Tensor::View bcast = Tensor::View::broadcast(scalar, {3, 5});

        std::cout << "broadcast shape: " << bcast.getShape().toString() << "\n";


        REQUIRE(bcast.getShape() == Tensor::Shape({3, 5}));
        REQUIRE(bcast.getStride() == std::vector<size_t>{0, 0});
    }
}

TEST_CASE("Copy of broadcast view gives repeated values", "[View][Stride]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor scalar(7.0f, backend);
        Tensor::View bcast = Tensor::View::broadcast(scalar, {4});

        Tensor result = bcast.copy();
        REQUIRE(result.getShape() == Tensor::Shape({4}));

        for (size_t i = 0; i < 4; i++) {
            REQUIRE(result[i] == Tensor(7.0f, backend));
        }
    }
}

// ---------------------------------------------------------------------------
// Arithmetic with strided views
// ---------------------------------------------------------------------------

TEST_CASE("Strided view + strided view", "[View][Stride][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({6}, 2.0f, backend);
        Tensor b({6}, 3.0f, backend);
        Tensor::View va = a.subsample({2});
        Tensor::View vb = b.subsample({2});

        Tensor result = va + vb;
        REQUIRE(result.getShape() == Tensor::Shape({3}));
        REQUIRE(result == Tensor({3}, 5.0f, backend));

        for (size_t i = 0; i < 3; i++) {
            REQUIRE(result[i] == Tensor(5.0f, backend));
        }
    }
}

TEST_CASE("Broadcast view + normal view", "[View][Stride][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor scalar(10.0f, backend);
        Tensor a({2, 4}, 3.0f, backend);

        Tensor::View bcast = Tensor::View::broadcast(scalar, {4});
        Tensor::View row = a[0];

        Tensor result = bcast + row;

        REQUIRE(result.getShape() == Tensor::Shape({4}));

        for (size_t j = 0; j < 4; j++) {
            REQUIRE(result[j] == Tensor(13.0f, backend));
        }
    }
}

TEST_CASE("Broadcast view * normal tensor", "[View][Stride][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor scalar(5.0f, backend);
        Tensor a({6}, 4.0f, backend);

        Tensor::View bcast = Tensor::View::broadcast(scalar, {6});

        Tensor result = bcast * a;

        for (size_t i = 0; i < 6; i++) {
            REQUIRE(result[i] == Tensor(20.0f, backend));
        }
    }
}

// ---------------------------------------------------------------------------
// Stride-related edge cases
// ---------------------------------------------------------------------------

TEST_CASE("Stride of 1 gives same shape as original", "[View][Stride]") {
    auto backend = GENERATE(from_range(backends));

    auto size = GENERATE(1ull, 4ull, 13ull);

    DYNAMIC_SECTION(getBackendString(backend) << " size=" << size) {
        Tensor a({size}, 2.0f, backend);
        Tensor::View b = a.subsample({1});

        REQUIRE(b.getShape() == Tensor::Shape({size}));

        Tensor c = b.copy();
        for (size_t i = 0; i < size; i++) {
            REQUIRE(c[i] == Tensor(2.0f, backend));
        }
    }
}

TEST_CASE("Stride equal to dimension gives single element", "[View][Stride]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({8}, 6.0f, backend);
        Tensor::View b = a.subsample({8});

        REQUIRE(b.getShape() == Tensor::Shape({1}));

        Tensor c = b.copy();
        REQUIRE(c[0] == Tensor(6.0f, backend));
    }
}

TEST_CASE("Mixed strides across dimensions", "[View][Stride]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({4, 6}, 1.0f, backend);
        // stride 2 on rows, stride 1 on cols
        Tensor::View b = a.subsample({2, 1});

        REQUIRE(b.getShape() == Tensor::Shape({2, 6}));

        b = Tensor({2, 6}, 8.0f, backend);

        // rows 0 and 2 should be written, rows 1 and 3 untouched
        for (size_t j = 0; j < 6; j++) {
            REQUIRE(a[0][j] == Tensor(8.0f, backend));
            REQUIRE(a[2][j] == Tensor(8.0f, backend));
            REQUIRE(a[1][j] == Tensor(1.0f, backend));
            REQUIRE(a[3][j] == Tensor(1.0f, backend));
        }
    }
}

TEST_CASE("Assign to zero-strided view multiple times", "[View][Stride]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({4}, 0.0f, backend);
        Tensor::View b = a.subsample({0});

        b = Tensor(1.0f, backend);
        REQUIRE(a[0] == Tensor(1.0f, backend));

        b = Tensor(2.0f, backend);
        REQUIRE(a[0] == Tensor(2.0f, backend));

        // only element 0 was ever touched
        REQUIRE(a[1] == Tensor(0.0f, backend));
        REQUIRE(a[2] == Tensor(0.0f, backend));
        REQUIRE(a[3] == Tensor(0.0f, backend));
    }
}

// ---------------------------------------------------------------------------
// Parametric stride tests
// ---------------------------------------------------------------------------

TEST_CASE("Parametric 1D stride consistency", "[View][Stride]") {
    auto backend = GENERATE(from_range(backends));

    auto total = GENERATE(6ull, 12ull, 24ull);
    auto stride = GENERATE(1ull, 2ull, 3ull);

    DYNAMIC_SECTION(
        "Backend=" << getBackendString(backend)
        << " total=" << total
        << " stride=" << stride
    ) {
        REQUIRE(total % stride == 0); // invalid test

        Tensor a({total}, 4.0f, backend);
        Tensor::View b = a.subsample({stride});

        REQUIRE(b.getShape() == Tensor::Shape({total / stride}));

        Tensor c = b.copy();
        for (size_t i = 0; i < total / stride; i++) {
            REQUIRE(c[i] == Tensor(4.0f, backend));
        }
    }
}