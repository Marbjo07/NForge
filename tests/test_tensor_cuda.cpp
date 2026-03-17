#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "nforge/core/tensor.h"

TEST_CASE("Create tensor", "[Tensor][CUDA]") {
    Tensor t({3}, 4.0f, Backend::CUDA);

    REQUIRE(t.getNumElements() == 3);
    REQUIRE(t.getBackendString() == "CUDA");
}

TEST_CASE("Compare tensor", "[Tensor][CUDA]") {
    Tensor a({3, 9, 7}, 19.0f, Backend::CUDA);
    Tensor b({3, 9, 7}, 19.0f, Backend::CUDA);
    Tensor c({3, 7, 9}, 19.0f, Backend::CUDA);

    REQUIRE(a == b);
    REQUIRE(b == a);
    REQUIRE(c != a);
    REQUIRE(c != b);
}

TEST_CASE("Compare tensor views", "[Tensor][CUDA]") {
    Tensor a({3, 9, 7}, 19.0f, Backend::CUDA);
    Tensor b({3, 9, 7}, 19.0f, Backend::CUDA);

    auto x = a[0];
    auto y = b[0];

    REQUIRE(x == y);
    REQUIRE(y == x);
}

TEST_CASE("Compare tensor and tensor view", "[Tensor][CUDA]") {
    Tensor a({3, 9, 7}, 19.0f, Backend::CUDA);
    Tensor b({9, 7}, 19.0f, Backend::CUDA);

    auto x = a[0];
    auto y = a[1];

    REQUIRE(x == b);
    REQUIRE(x == y);

    REQUIRE(b == x);
    REQUIRE(b == y);

    REQUIRE(y == x);
    REQUIRE(y == b);
}

TEST_CASE("Add tensors", "[Tensor][CUDA]") {
    Tensor a({3}, 4.0f, Backend::CUDA);
    Tensor b({3}, 1.0f, Backend::CUDA);

    Tensor c = a + b;

    REQUIRE(c[0] == Tensor(5.0f, Backend::CUDA));
    REQUIRE(c[0] == c[1]);
    REQUIRE(c[1] == c[2]);
    REQUIRE(c[0] != Tensor(3.0f, Backend::CUDA));
}

TEST_CASE("Subtract and multiply tensors", "[Tensor][CUDA]") {
    Tensor a({3}, 4.0f, Backend::CUDA);
    Tensor b({3}, 1.0f, Backend::CUDA);

    Tensor sub = a - b;
    Tensor mul = a * b;

    REQUIRE(sub[0] == Tensor(3.0f, Backend::CUDA));
    REQUIRE(sub[0] == sub[1]);
    REQUIRE(sub[1] == sub[2]);

    REQUIRE(mul[0] == Tensor(4.0f, Backend::CUDA));
    REQUIRE(mul[0] == mul[1]);
    REQUIRE(mul[1] == mul[2]);
}

TEST_CASE("Broadcast scalar add", "[Tensor][CUDA]") {
    Tensor a({4}, 2.0f, Backend::CUDA);
    Tensor s(3.0f, Backend::CUDA);

    Tensor x = a + s;
    Tensor y = s + a;

	INFO("a=" + a.getDataString());
	INFO("s=" + s.getDataString());
	INFO("x=" + x.getDataString());
	INFO("y=" + y.getDataString());

	for (size_t i = 0; i < 4; i++) {
		REQUIRE(x[i] == Tensor(5.0f, Backend::CUDA));
    	REQUIRE(y[i] == Tensor(5.0f, Backend::CUDA));
	}
}

TEST_CASE("2D tensor shape and indexing", "[Tensor][CUDA]") {
    auto rows = GENERATE(1ull, 4ull, 10ull);
    auto cols = GENERATE(1ull, 10ull, 50ull);
    auto val = GENERATE(-1001.0f, 0.32f, 122.9f);

    DYNAMIC_SECTION("rows=" << rows << " cols=" << cols << " val=" << val) {
        Tensor a({rows, cols}, val, Backend::CUDA);

        // Check number of elements
        REQUIRE(a.getNumElements() == rows * cols);

        // Compare slices
        REQUIRE(a[0] == Tensor({cols}, val, Backend::CUDA));
        REQUIRE(a[0][0] == Tensor(val, Backend::CUDA));
        REQUIRE(a[0][0] != Tensor(val - 1, Backend::CUDA));

        REQUIRE_FALSE(a[0][0] != Tensor(val, Backend::CUDA));
        REQUIRE_FALSE(a[0][0] == Tensor(val - 1, Backend::CUDA));

        REQUIRE(a[0] == a[rows - 1]);
    }
}

TEST_CASE("Tensor slice assign", "[Tensor][CUDA]") {
    auto rows = GENERATE(1ull, 2ull, 3ull);
    auto cols = GENERATE(1ull, 4ull, 8ull);
    auto val = GENERATE(0.0f, 1.5f);

    DYNAMIC_SECTION("rows=" << rows << " cols=" << cols << " val=" << val) {
        // Create A and random B
        Tensor A({rows, cols}, val, Backend::CUDA);
        Tensor B({rows, cols}, Backend::CUDA);
        B.fillRand();

        // Slice copy
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