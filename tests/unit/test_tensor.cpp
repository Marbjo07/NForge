#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "nforge/nforge.h"
#include "utils.h"


TEST_CASE("Create tensor", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor t({3}, 4.0f, backend);

        REQUIRE(t.getNumElements() == 3);
    }
}

TEST_CASE("Compare tensor", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({3, 9, 7}, 19.0f, backend);
        Tensor b({3, 9, 7}, 19.0f, backend);
        Tensor c({3, 7, 9}, 19.0f, backend);

        REQUIRE(a == b);
        REQUIRE(b == a);
        REQUIRE(c != a);
        REQUIRE(c != b);
    }
}

TEST_CASE("Compare tensor views", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({3, 9, 7}, 19.0f, backend);
        Tensor b({3, 9, 7}, 19.0f, backend);

        auto x = a[0];
        auto y = b[0];

        REQUIRE(x == y);
        REQUIRE(y == x);
    }
}

TEST_CASE("Compare tensor and tensor view", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({3, 9, 7}, 19.0f, backend);
        Tensor b({9, 7}, 0.0f, backend);

        auto x = a[0];
        auto y = a[1];

        REQUIRE(x != b);
        REQUIRE(x == y);

        REQUIRE(b != x);
        REQUIRE(b != y);

        REQUIRE(y == x);
        REQUIRE(y != b);
    }
}

TEST_CASE("Tensor view copy and compare", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({3, 9, 7}, 19.0f, backend);
        Tensor b({9, 7}, 0.0f, backend);
        Tensor c({3, 9, 7}, 19.0f, backend);

        a[0] = b;
        a[1] = b;

        REQUIRE(a != c);
        REQUIRE(a[0] != c[0]);
        REQUIRE(a[2] == c[1]);

        REQUIRE(c != a);
        REQUIRE(c[0] != a[0]);
        REQUIRE(c[0] == a[2]);


        REQUIRE(a[1] == b);
        REQUIRE(a[2] != b);

        REQUIRE(b == a[1]);
        REQUIRE(b != a[2]);
    }
}


TEST_CASE("Add tensors", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({3}, 4.0f, backend);
        Tensor b({3}, 1.0f, backend);

        Tensor c = a + b;

        REQUIRE(c[0] == Tensor(5.0f, backend));
        REQUIRE(c[0] == c[1]);
        REQUIRE(c[1] == c[2]);
        REQUIRE(c[0] != Tensor(3.0f, backend));
    }
}

TEST_CASE("Subtract and multiply tensors", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({3}, 4.0f, backend);
        Tensor b({3}, 1.0f, backend);

        Tensor sub = a - b;
        Tensor mul = a * b;

        REQUIRE(sub[0] == Tensor(3.0f, backend));
        REQUIRE(sub[0] == sub[1]);
        REQUIRE(sub[1] == sub[2]);

        REQUIRE(mul[0] == Tensor(4.0f, backend));
        REQUIRE(mul[0] == mul[1]);
        REQUIRE(mul[1] == mul[2]);
    }
}

TEST_CASE("Broadcast scalar add", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({4}, 2.0f, backend);
        Tensor s(3.0f, backend);
        
        Tensor x = a + s;
        Tensor y = s + a;

        for (size_t i = 0; i < 4; i++) {
            REQUIRE(x[i] == Tensor(5.0f, backend));
            REQUIRE(y[i] == Tensor(5.0f, backend));
        }
    }
}

TEST_CASE("2D tensor shape and indexing", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        auto rows = GENERATE(1ull, 4ull, 10ull);
        auto cols = GENERATE(1ull, 10ull, 50ull);
        auto val = GENERATE(-1001.0f, 0.32f, 122.9f);

        DYNAMIC_SECTION("rows=" << rows << " cols=" << cols << " val=" << val) {
            Tensor a({rows, cols}, val, backend);

            // Check number of elements
            REQUIRE(a.getNumElements() == rows * cols);

            // Compare views
            REQUIRE(a[0] == Tensor({cols}, val, backend));
            REQUIRE(a[0][0] == Tensor(val, backend));
            REQUIRE(a[0][0] != Tensor(val - 1, backend));

            REQUIRE_FALSE(a[0][0] != Tensor(val, backend));
            REQUIRE_FALSE(a[0][0] == Tensor(val - 1, backend));

            REQUIRE(a[0] == a[rows - 1]);
        }
    }
}

TEST_CASE("Chained tensor assignment", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({3}, 1.0f, backend);
        Tensor b({3}, 2.0f, backend);
        Tensor c({3}, 3.0f, backend);

        // a = b = c should make both a and b equal to c
        a = b = c;

        REQUIRE(b == c);
        REQUIRE(a == c);
        REQUIRE(a == b);
    }
}


TEST_CASE("Tensor view assign", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        auto rows = GENERATE(1ull, 2ull, 3ull);
        auto cols = GENERATE(1ull, 4ull, 8ull);
        auto val = GENERATE(0.0f, 1.5f);

        DYNAMIC_SECTION("rows=" << rows << " cols=" << cols << " val=" << val) {
            // Create A and random B
            Tensor A({rows, cols}, val, backend);
            Tensor B({rows, cols}, backend);
            B.fillRand();

            // View copy
            for (size_t i = 0; i < rows; i++) {
                A[i] = B[i];
            }

            // Element wise compare
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    REQUIRE(A[i][j] == B[i][j]);
                }
            }

            REQUIRE(A == B);
        }
    }
}

TEST_CASE("Tensor and float add", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({4}, 2.0f, backend);
        Tensor b = a + 3;

        REQUIRE(b == Tensor({4}, 5.0f, backend));
    }
}


TEST_CASE("Tensor and float sub", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({4}, 2.0f, backend);
        Tensor b = a - 3;

        REQUIRE(b == Tensor({4}, -1.0f, backend));
    }
}


TEST_CASE("Tensor and float mul", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({4}, 2.0f, backend);
        Tensor b = a * 3;

        REQUIRE(b == Tensor({4}, 6.0f, backend));
    }
}


TEST_CASE("Tensor and float div", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({4}, 2.0f, backend);
        Tensor b = a / 3;

        REQUIRE(b == Tensor({4}, 2.0f / 3.0f, backend));
    }
}


TEST_CASE("Float and Tensor add", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({4}, 2.0f, backend);
        Tensor b = 3 + a;

        REQUIRE(b == Tensor({4}, 5.0f, backend));
    }
}


TEST_CASE("Float and Tensor sub", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({4}, 2.0f, backend);
        Tensor b = 3 - a;

        REQUIRE(b == Tensor({4}, 1.0f, backend));
    }
}


TEST_CASE("Float and Tensor mul", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({4}, 2.0f, backend);
        Tensor b = 3 * a;

        REQUIRE(b == Tensor({4}, 6.0f, backend));
    }
}


TEST_CASE("Float and Tensor div", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({4}, 2.0f, backend);
        Tensor b = 3 / a;

        REQUIRE(b == Tensor({4}, 1.5f, backend));
    }
}

TEST_CASE("Invalid Tensor index throws", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({2}, 2.0f, backend);
        REQUIRE_THROWS(a[3]);
    }
}

TEST_CASE("Invalid Tensor 2D index throws", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({2, 6}, 2.0f, backend);

        REQUIRE_THROWS(a[0][6]);
        REQUIRE_THROWS(a[2][4]);
        REQUIRE_THROWS(a[1][6]);
        REQUIRE_THROWS(a[-1][0]);
        REQUIRE_THROWS(a[-1][-1]);
    }
}

TEST_CASE("Scalar Tensor initialization", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        for (size_t n : {1, 5, 8, 16}) {
            Tensor a({n}, backend);
            REQUIRE(a.getShape() == Tensor::Shape({n}));

            Tensor b({n}, 1.0f, backend);
            REQUIRE(a.getShape() == Tensor::Shape({n}));
        }

        Tensor a(1.0f, backend);
        REQUIRE(a.getShape() == Tensor::Shape({1}));
    }
}

TEST_CASE("Frobenius norm of scalar", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a(4.0f, backend);
        Tensor b(-4.0f, backend);

        REQUIRE(a.norm() == 4.0f);
        REQUIRE(b.norm() == 4.0f);
    }
}


TEST_CASE("Frobenius norm of vector", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({5}, -4.0f, backend);

        REQUIRE(a.norm() == 4.0f);
        
        for (size_t i = 0; i < 5; i++) {
            a[i] = i;
        }

        // a = [0, 1, 2, 3, 4]
        REQUIRE(a.norm() == std::sqrt(30.0f));
    }
}


TEST_CASE("Frobenius norm of matrix", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        size_t n = 5;
        size_t m = 8;
        Tensor a({n, m}, -4.0f, backend);

        REQUIRE(a.norm() == 4.0f);
        
        float sum = 0;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < m; j++) {
                a[i][j] = i * m + j;
                sum += i * m + j;
            }
        }
        
        REQUIRE(a.norm() == std::sqrt(sum));
    }
}

TEST_CASE("Frobenius norm of random tensor", "[Tensor]") {
    auto backend = GENERATE(from_range(backends));

    DYNAMIC_SECTION(getBackendString(backend)) {
        Tensor a({8, 9, 4, 3}, -4.0f, backend);

        REQUIRE(a.norm() == 4.0f);
        
        a.fillRand();

        double sum = 0;
        const auto elements = a.toVector();
        for (float e : elements) {
            sum += e;
        }
        
        REQUIRE(a.norm() == std::sqrt(sum));
    }
}