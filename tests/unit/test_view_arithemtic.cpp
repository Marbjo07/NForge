// =============================================================================
// Tests for: Tensor and Tensor View arithmetic operations
//
// Covers:
//   1. View::copy() - converting a view back to an owned Tensor
//   2. View <op> View arithmetic
//   3. Tensor <op> View  and  View <op> Tensor arithmetic
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "nforge/nforge.h"

#include "utils.h"


// ---------------------------------------------------------------------------
// 1. View::copy()
// ---------------------------------------------------------------------------

TEST_CASE("View copy produces an independent Tensor", "[View]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor src({3, 4}, 7.0f, backend);

        auto   view = src[1];              // view of row-1 (4 elements)
        Tensor copy = view.copy();

        // Dimensions match the viewed slice
        REQUIRE(copy.getShape() == Tensor::Shape({4}));

        // Values match
        for (size_t j = 0; j < 4; j++) {
            REQUIRE(copy[j] == Tensor(7.0f, backend));
        }
    }
}

TEST_CASE("View copy is a deep copy", "[View]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor src({4}, 1.0f, backend);
        auto   view = src[0];             // scalar view
        Tensor copy = view.copy();

        // Mutate the source and copy must be unaffected
        src = Tensor({4}, 99.0f, backend);
        REQUIRE(copy == Tensor(1.0f, backend));
    }
}

TEST_CASE("Copy of a 2D sub-view preserves values", "[View]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor src({3, 5}, 4.0f, backend);
        Tensor row2 = src[2].copy();

        REQUIRE(row2.getShape() == Tensor::Shape({5}));

        for (size_t j = 0; j < 5; j++) {
            REQUIRE(row2[j] == Tensor(4.0f, backend));
        }
    }
}

// ---------------------------------------------------------------------------
// 2. View + View arithmetic
// ---------------------------------------------------------------------------

TEST_CASE("View + View addition", "[View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor A({2, 4}, 3.0f, backend);
        Tensor B({2, 4}, 5.0f, backend);

        // Views into row-0 of each tensor
        auto va = A[0];
        auto vb = B[0];

        Tensor result = va + vb;

        REQUIRE(result.getShape() == Tensor::Shape({4}));

        for (size_t j = 0; j < 4; j++) {
            REQUIRE(result[j] == Tensor(8.0f, backend));
        }
    }
}

TEST_CASE("View - View subtraction", "[View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor A({2, 4}, 10.0f, backend);
        Tensor B({2, 4}, 3.0f,  backend);

        Tensor result = A[0] - B[0];

        for (size_t j = 0; j < 4; j++) {
            REQUIRE(result[j] == Tensor(7.0f, backend));
        }
    }
}

TEST_CASE("View * View multiplication", "[View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor A({2, 4}, 4.0f, backend);
        Tensor B({2, 4}, 2.5f, backend);

        Tensor result = A[1] * B[1];

        for (size_t j = 0; j < 4; j++) {
            REQUIRE(result[j] == Tensor(10.0f, backend));
        }
    }
}

TEST_CASE("View / View division", "[View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor A({2, 4}, 9.0f, backend);
        Tensor B({2, 4}, 3.0f, backend);

        Tensor result = A[0] / B[0];

        for (size_t j = 0; j < 4; j++) {
            REQUIRE(result[j] == Tensor(3.0f, backend));
        }
    }
}

TEST_CASE("View-View arithmetic with different rows of the same Tensor", "[View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor M({3, 5}, 0.0f, backend);
        Tensor row_vals({3, 5}, 1.0f, backend);
        Tensor T = row_vals + row_vals; // all 2.0

        // row0 + row2 of the same tensor
        Tensor sum = T[0] + T[2];

        for (size_t j = 0; j < 5; j++) {
            REQUIRE(sum[j] == Tensor(4.0f, backend));
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Tensor <op> View  and  View <op> Tensor
// ---------------------------------------------------------------------------

TEST_CASE("Tensor + View", "[Tensor][View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor A({4}, 2.0f, backend);
        Tensor B({2, 4}, 3.0f, backend);

        Tensor result = A + B[0];

        REQUIRE(result.getShape() == Tensor::Shape({4}));

        for (size_t j = 0; j < 4; j++) {
            REQUIRE(result[j] == Tensor(5.0f, backend));
        }
    }
}

TEST_CASE("View + Tensor", "[Tensor][View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor A({2, 4}, 6.0f, backend);
        Tensor B({4}, 1.0f, backend);

        Tensor result = A[1] + B;

        for (size_t j = 0; j < 4; j++) {
            REQUIRE(result[j] == Tensor(7.0f, backend));
        }
    }
}

TEST_CASE("Tensor - View", "[Tensor][View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor A({4}, 10.0f, backend);
        Tensor B({2, 4}, 4.0f, backend);

        Tensor result = A - B[0];

        for (size_t j = 0; j < 4; j++) {
            REQUIRE(result[j] == Tensor(6.0f, backend));
        }
    }
}

TEST_CASE("View * Tensor", "[Tensor][View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor A({2, 4}, 3.0f, backend);
        Tensor B({4}, 5.0f, backend);

        Tensor result = A[0] * B;

        for (size_t j = 0; j < 4; j++) {
            REQUIRE(result[j] == Tensor(15.0f, backend));
        }
    }
}

TEST_CASE("View / Tensor", "[Tensor][View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor A({2, 4}, 12.0f, backend);
        Tensor B({4}, 4.0f, backend);

        Tensor result = A[1] / B;

        for (size_t j = 0; j < 4; j++) {
            REQUIRE(result[j] == Tensor(3.0f, backend));
        }
    }
}

// ---------------------------------------------------------------------------
// 3b. Scalar broadcasting with views
// ---------------------------------------------------------------------------

TEST_CASE("Scalar Tensor + View", "[Tensor][View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor scalar(10.0f, backend);
        Tensor M({2, 3}, 2.0f, backend);

        Tensor result = scalar + M[0];

        for (size_t j = 0; j < 3; j++) {
            REQUIRE(result[j] == Tensor(12.0f, backend));
        }
    }
}

TEST_CASE("View + scalar Tensor", "[Tensor][View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor M({2, 3}, 5.0f, backend);
        Tensor scalar(1.0f, backend);

        Tensor result = M[1] + scalar;

        for (size_t j = 0; j < 3; j++) {
            REQUIRE(result[j] == Tensor(6.0f, backend));
        }
    }
}

TEST_CASE("View * scalar Tensor", "[Tensor][View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor M({2, 4}, 3.0f, backend);
        Tensor scalar(4.0f, backend);

        Tensor result = M[0] * scalar;

        for (size_t j = 0; j < 4; j++) {
            REQUIRE(result[j] == Tensor(12.0f, backend));
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Parametric 2D consistency
// ---------------------------------------------------------------------------

TEST_CASE("2D View arithmetic consistency across backends", "[View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    auto rows = GENERATE(2ul, 4ul, 8ul);
    auto cols = GENERATE(1ul, 5ul, 11ul);

    DYNAMIC_SECTION(
        "Backend=" << getBackendString(backend)
        << " rows=" << rows
        << " cols=" << cols
    ) {
        Tensor A({rows, cols}, 6.0f, backend);
        Tensor B({rows, cols}, 2.0f, backend);

        for (size_t i = 0; i < rows; i++) {
            Tensor sum  = A[i] + B[i];
            Tensor diff = A[i] - B[i];
            Tensor prod = A[i] * B[i];
            Tensor quot = A[i] / B[i];

            for (size_t j = 0; j < cols; j++) {
                REQUIRE(sum[j]  == Tensor(8.0f,  backend));
                REQUIRE(diff[j] == Tensor(4.0f,  backend));
                REQUIRE(prod[j] == Tensor(12.0f, backend));
                REQUIRE(quot[j] == Tensor(3.0f,  backend));
            }
        }
    }
}

TEST_CASE("Mixed Tensor/View 2D parametric test", "[Tensor][View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    auto rows = GENERATE(1ul, 3ul, 7ul);
    auto cols = GENERATE(1ul, 5ul, 11ul);

    DYNAMIC_SECTION(
        "Backend=" << getBackendString(backend)
        << " rows=" << rows
        << " cols=" << cols
    ) {
        Tensor A({rows, cols}, 4.0f, backend);
        Tensor vec({cols}, 1.0f, backend);

        for (size_t i = 0; i < rows; i++) {
            Tensor sum  = A[i] + vec;     // view + tensor
            Tensor prod = vec * A[i];     // tensor * view

            for (size_t j = 0; j < cols; j++) {
                REQUIRE(sum[j]  == Tensor(5.0f, backend));
                REQUIRE(prod[j] == Tensor(4.0f, backend));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Edge cases
// ---------------------------------------------------------------------------

TEST_CASE("Arithmetic on single-element views", "[View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor A({1, 1}, 5.0f, backend);
        Tensor B({1, 1}, 3.0f, backend);

        Tensor result = A[0] + B[0];
        REQUIRE(result[0] == Tensor(8.0f, backend));
    }
}

TEST_CASE("Chained view arithmetic expressions", "[View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor A({2, 4}, 2.0f, backend);
        Tensor B({2, 4}, 3.0f, backend);
        Tensor C({2, 4}, 4.0f, backend);

        // (A[0] + B[0]) * C[0] => (2+3)*4 = 20
        Tensor result = (A[0] + B[0]) * C[0];

        for (size_t j = 0; j < 4; j++) {
            REQUIRE(result[j] == Tensor(20.0f, backend));
        }

        // A, B, C should remain unchanged
        REQUIRE(A == Tensor({2, 4}, 2.0f, backend));
        REQUIRE(B == Tensor({2, 4}, 3.0f, backend));
        REQUIRE(C == Tensor({2, 4}, 4.0f, backend));
    }
}

TEST_CASE("Copy then arithmetic gives correct result", "[View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor M({3, 4}, 5.0f, backend);

        Tensor row = M[1].copy();
        Tensor other({4}, 2.0f, backend);

        Tensor result = row + other;

        for (size_t j = 0; j < 4; j++) {
            REQUIRE(result[j] == Tensor(7.0f, backend));
        }
    }
}

TEST_CASE("Arithmetic result does not alias the source view", "[View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor A({2, 4}, 1.0f, backend);
        Tensor B({2, 4}, 2.0f, backend);

        Tensor sum = A[0] + B[0]; // should be 3.0

        // Mutate A - sum must stay 3.0
        A = Tensor({2, 4}, 99.0f, backend);

        for (size_t j = 0; j < 4; j++) {
            REQUIRE(sum[j] == Tensor(3.0f, backend));
        }
    }
}

TEST_CASE("In-place addition operator on view", "[View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({4, 3}, 3.0f, backend);
        Tensor b({3}, 2.0f, backend);
        
        for (size_t d = 0; d < 4; d++) {
            a[d] += b;
            REQUIRE(a[d] == Tensor({3}, 5.0f, backend));
        }
    }
}

TEST_CASE("In-place subtraction operator on view", "[View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({4, 3}, 3.0f, backend);
        Tensor b({3}, 2.0f, backend);
        
        for (size_t d = 0; d < 4; d++) {
            a[d] -= b;
            REQUIRE(a[d] == Tensor({3}, 1.0f, backend));
        }
    }
}

TEST_CASE("In-place multiplication operator on view", "[View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({4, 3}, 3.0f, backend);
        Tensor b({3}, 2.0f, backend);
        
        for (size_t d = 0; d < 4; d++) {
            a[d] *= b;
            REQUIRE(a[d] == Tensor({3}, 6.0f, backend));
        }
    }
}

TEST_CASE("In-place division operator on view", "[View][Arithmetic]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({4, 3}, 3.0f, backend);
        Tensor b({3}, 2.0f, backend);
        
        for (size_t d = 0; d < 4; d++) {
            a[d] /= b;
            REQUIRE(a[d] == Tensor({3}, 1.5f, backend));
        }
    }
}