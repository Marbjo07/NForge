#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "nforge/nforge.h"
#include "utils.h"

// TODO: refactor with tensor construction from vector
Tensor makeVectorTensor(std::vector<float> values, Backend backend) {
	Tensor t({values.size()}, backend);
	for (size_t i = 0; i < values.size(); i++) {
		t[i] = Tensor(values[i], backend);
	}
	return t;
}


/*
   a = [[0, 1, 2],
        [3, 4, 5]]
*/

TEST_CASE("Tensor reductions", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));
	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({2, 3}, backend);
		for (size_t i = 0; i < 2; i++) {
			for (size_t j = 0; j < 3; j++) {
				a[i][j] = Tensor(i * 3 + j, backend);
			}
		}


		SECTION("sum") {
			REQUIRE(tensor_equal(a.sum(), Tensor(15.0f, backend)));
			REQUIRE(tensor_equal(a.sum(0), Tensor(15.0f, backend)));
			REQUIRE(tensor_equal(a.sum(1), makeVectorTensor({3.0f, 12.0f}, backend)));
			REQUIRE(tensor_equal(a.sum(2), a));
		}

		SECTION("mean") {
			REQUIRE(tensor_equal(a.mean(), Tensor(2.5f, backend)));
			REQUIRE(tensor_equal(a.mean(0), Tensor(2.5f, backend)));
			REQUIRE(tensor_equal(a.mean(1), makeVectorTensor({1.0f, 4.0f}, backend)));
			REQUIRE(tensor_equal(a.mean(2), a));
		}

		SECTION("max") {
			REQUIRE(tensor_equal(a.max(), Tensor(5.0f, backend)));
			REQUIRE(tensor_equal(a.max(0), Tensor(5.0f, backend)));
			REQUIRE(tensor_equal(a.max(1), makeVectorTensor({2.0f, 5.0f}, backend)));
			REQUIRE(tensor_equal(a.max(2), a));
		}

		SECTION("min") {
			REQUIRE(tensor_equal(a.min(), Tensor(0.0f, backend)));
			REQUIRE(tensor_equal(a.min(0), Tensor(0.0f, backend)));
			REQUIRE(tensor_equal(a.min(1), makeVectorTensor({0.0f, 3.0f}, backend)));
			REQUIRE(tensor_equal(a.min(2), a));
		}

		SECTION("prod") {
			REQUIRE(tensor_equal(a.prod(), Tensor(0.0f, backend)));
			REQUIRE(tensor_equal(a.prod(0), Tensor(0.0f, backend)));
			REQUIRE(tensor_equal(a.prod(1), makeVectorTensor({0.0f, 60.0f}, backend)));
			REQUIRE(tensor_equal(a.prod(2), a));
		}
	}
}


TEST_CASE("View reductions", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));
	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor orig({3, 2, 3}, backend);

		Tensor::View a = orig[2];
		for (size_t i = 0; i < 2; i++) {
			for (size_t j = 0; j < 3; j++) {
				a[i][j] = Tensor(i * 3 + j, backend);
			}
		}


		SECTION("sum") {
			REQUIRE(tensor_equal(a.sum(), Tensor(15.0f, backend)));
			REQUIRE(tensor_equal(a.sum(0), Tensor(15.0f, backend)));
			REQUIRE(tensor_equal(a.sum(1), makeVectorTensor({3.0f, 12.0f}, backend)));
			REQUIRE(tensor_equal(a.sum(2), a));
		}

		SECTION("mean") {
			REQUIRE(tensor_equal(a.mean(), Tensor(2.5f, backend)));
			REQUIRE(tensor_equal(a.mean(0), Tensor(2.5f, backend)));
			REQUIRE(tensor_equal(a.mean(1), makeVectorTensor({1.0f, 4.0f}, backend)));
			REQUIRE(tensor_equal(a.mean(2), a));
		}

		SECTION("max") {
			REQUIRE(tensor_equal(a.max(), Tensor(5.0f, backend)));
			REQUIRE(tensor_equal(a.max(0), Tensor(5.0f, backend)));
			REQUIRE(tensor_equal(a.max(1), makeVectorTensor({2.0f, 5.0f}, backend)));
			REQUIRE(tensor_equal(a.max(2), a));
		}

		SECTION("min") {
			REQUIRE(tensor_equal(a.min(), Tensor(0.0f, backend)));
			REQUIRE(tensor_equal(a.min(0), Tensor(0.0f, backend)));
			REQUIRE(tensor_equal(a.min(1), makeVectorTensor({0.0f, 3.0f}, backend)));
			REQUIRE(tensor_equal(a.min(2), a));
		}

		SECTION("prod") {
			REQUIRE(tensor_equal(a.prod(), Tensor(0.0f, backend)));
			REQUIRE(tensor_equal(a.prod(0), Tensor(0.0f, backend)));
			REQUIRE(tensor_equal(a.prod(1), makeVectorTensor({0.0f, 60.0f}, backend)));
			REQUIRE(tensor_equal(a.prod(2), a));
		}
	}
}


TEST_CASE("Tensor mean equals sum divided by count", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));
	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({4, 5, 8}, backend);
		a.fillRand();

		size_t count = 4 * 5 * 8;
		for (size_t d = 0; d <= 3; d++) {
			Tensor diff = a.mean(d) - a.sum(d) / count;
			REQUIRE((diff * diff).max().toVector()[0] < 1e-6f);

			if (d != 3) {
				count /= a.getShape().getDim(d);
			}
		}
	}
}

/*

   a = [[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23]]

        subsampled {2,2}

    b = [[ 0,  2,  4],
         [12, 14, 16]]

*/
TEST_CASE("Tensor reductions respect strides", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));
	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({4, 6}, backend);
		for (size_t i = 0; i < 4; i++) {
			for (size_t j = 0; j < 6; j++) {
				a[i][j] = Tensor(i * 6 + j, backend);
			}
		}

		Tensor::View b = a.subsample({2, 2});

		SECTION("mean") {
			REQUIRE(tensor_equal(b.mean(0), Tensor(8.0f, backend)));
			REQUIRE(tensor_equal(b.mean(1), makeVectorTensor({2.0f, 14.0f}, backend)));
		}

		SECTION("sum") {
			REQUIRE(tensor_equal(b.sum(0), Tensor(48.0f, backend)));
			REQUIRE(tensor_equal(b.sum(1), makeVectorTensor({6.0f, 42.0f}, backend)));
		}

		SECTION("max") {
			REQUIRE(tensor_equal(b.max(0), Tensor(16.0f, backend)));
			REQUIRE(tensor_equal(b.max(1), makeVectorTensor({4.0f, 16.0f}, backend)));
		}

		SECTION("min") {
			REQUIRE(tensor_equal(b.min(0), Tensor(0.0f, backend)));
			REQUIRE(tensor_equal(b.min(1), makeVectorTensor({0.0f, 12.0f}, backend)));
		}

		SECTION("prod") {
			REQUIRE(tensor_equal(b.prod(0), Tensor(0.0f, backend)));
			REQUIRE(
			    tensor_equal(b.prod(1), makeVectorTensor({0.0f, 12.0f * 14.0f * 16.0f}, backend)));
		}
	}
}

TEST_CASE("Tensor all reduction", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));
	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({4, 6}, backend);
		for (size_t i = 0; i < 4; i++) {
			for (size_t j = 0; j < 6; j++) {
				float value = (i * 6 + j + 4) % 7;
				a[i][j] = Tensor(value, backend);
			}
		}
		// a = [[4, 5, 6, 0, 1, 2],
		//      [3, 4, 5, 6, 0, 1],
		//      [2, 3, 4, 5, 6, 0],
		//      [1, 2, 3, 4, 5, 6]]


		std::vector<float> expected = {1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f,
		                               1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
		                               1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

		REQUIRE(tensor_equal(a.all(), Tensor(0.0f, backend)));
		REQUIRE(tensor_equal(a.all(0), Tensor(0.0f, backend)));
		REQUIRE(tensor_equal(a.all(1), makeVectorTensor({0.0f, 0.0f, 0.0f, 1.0f}, backend)));
		REQUIRE(a.all(2).toVector() == expected);
	}
}


TEST_CASE("Tensor any reduction", "[Tensor]") {
	auto backend = GENERATE(from_range(backends));
	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({4, 6}, backend);
		for (size_t i = 0; i < 4; i++) {
			for (size_t j = 0; j < 6; j++) {
				float value = (i != 2) * ((i * 6 + j) % 5);
				a[i][j] = Tensor(value, backend);
			}
		}
		// a = [[0, 1, 2, 3, 4, 0],
		//      [1, 2, 3, 4, 0, 1],
		// 	    [0, 0, 0, 0, 0, 0],
		//      [3, 4, 0, 1, 2, 3]]

		std::vector<float> expected = {0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
		                               1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		                               0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f};

		REQUIRE(tensor_equal(a.any(), Tensor(1.0f, backend)));
		REQUIRE(tensor_equal(a.any(0), Tensor(1.0f, backend)));
		REQUIRE(tensor_equal(a.any(1), makeVectorTensor({1.0f, 1.0f, 0.0f, 1.0f}, backend)));
		REQUIRE(a.any(2).toVector() == expected);
	}
}