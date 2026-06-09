#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "nforge/nforge.h"
#include "utils.h"

TEST_CASE("to() same backend is no-op", "[Tensor][to]") {
	auto backend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(backend)) {
		Tensor a({3, 4, 5}, backend);
		a.fillRand();

		auto original = a.toVector();
		auto before = a.getBackend();

		a.to(backend);

		REQUIRE(a.getBackend() == before);
		REQUIRE(a.toVector() == original);
	}
}

TEST_CASE("to() preserves data CPU->CPU", "[Tensor][to]") {
	Tensor a({2, 3}, 42.0f, Backend::CPU);
	Tensor b = a;

	a.to(Backend::CPU);

	REQUIRE(a.getBackend() == Backend::CPU);
	REQUIRE(a == b);
}

TEST_CASE("to() preserves data with random values CPU", "[Tensor][to]") {
	Tensor a({8, 6, 4}, Backend::CPU);
	a.fillRand();

	auto original = a.toVector();

	a.to(Backend::CPU);

	REQUIRE(a.getBackend() == Backend::CPU);
	REQUIRE(a.toVector() == original);
}

TEST_CASE("to() on scalar tensor", "[Tensor][to]") {
	Tensor a(3.14f, Backend::CPU);

	a.to(Backend::CPU);

	REQUIRE(a.getBackend() == Backend::CPU);
	REQUIRE(a == Tensor(3.14f, Backend::CPU));
}

TEST_CASE("to() on 1D tensor", "[Tensor][to]") {
	Tensor a({10}, Backend::CPU);
	for (size_t i = 0; i < 10; i++) {
		a[i] = static_cast<float>(i * i);
	}

	a.to(Backend::CPU);

	REQUIRE(a.getBackend() == Backend::CPU);
	for (size_t i = 0; i < 10; i++) {
		REQUIRE(a[i] == Tensor(static_cast<float>(i * i), Backend::CPU));
	}
}

TEST_CASE("to() on large 1D tensor", "[Tensor][to]") {
	size_t n = 1000;
	Tensor a({n}, Backend::CPU);
	for (size_t i = 0; i < n; i++) {
		a[i] = static_cast<float>(i);
	}

	a.to(Backend::CPU);

	REQUIRE(a.getBackend() == Backend::CPU);
	auto vec = a.toVector();
	REQUIRE(vec.size() == n);
	for (size_t i = 0; i < n; i++) {
		REQUIRE(vec[i] == static_cast<float>(i));
	}
}

#ifdef NFORGE_WITH_CUDA

TEST_CASE("to() transfers CPU->CUDA preserves data", "[Tensor][to][CUDA]") {
	Tensor a({3, 4, 5}, Backend::CPU);
	a.fillRand();

	auto original = a.toVector();

	a.to(Backend::CUDA);

	REQUIRE(a.getBackend() == Backend::CUDA);
	REQUIRE(a.toVector() == original);
}

TEST_CASE("to() transfers CUDA->CPU preserves data", "[Tensor][to][CUDA]") {
	Tensor a({3, 4, 5}, Backend::CUDA);
	a.fillRand();

	auto original = a.toVector();

	a.to(Backend::CPU);

	REQUIRE(a.getBackend() == Backend::CPU);
	REQUIRE(a.toVector() == original);
}

TEST_CASE("to() round-trip CPU->CUDA->CPU preserves data", "[Tensor][to][CUDA]") {
	Tensor a({6, 7}, 3.14f, Backend::CPU);

	auto original = a.toVector();

	a.to(Backend::CUDA);
	a.to(Backend::CPU);

	REQUIRE(a.getBackend() == Backend::CPU);
	REQUIRE(a.toVector() == original);
}

TEST_CASE("to() same backend CUDA is no-op", "[Tensor][to][CUDA]") {
	Tensor a({3, 4, 5}, Backend::CUDA);
	a.fillRand();

	auto original = a.toVector();

	a.to(Backend::CUDA);

	REQUIRE(a.getBackend() == Backend::CUDA);
	REQUIRE(a.toVector() == original);
}

TEST_CASE("to() CPU->CUDA on scalar", "[Tensor][to][CUDA]") {
	Tensor a(2.718f, Backend::CPU);

	a.to(Backend::CUDA);

	REQUIRE(a.getBackend() == Backend::CUDA);
	REQUIRE(a == Tensor(2.718f, Backend::CUDA));
}

TEST_CASE("to() CUDA->CPU on scalar", "[Tensor][to][CUDA]") {
	Tensor a(2.718f, Backend::CUDA);

	a.to(Backend::CPU);

	REQUIRE(a.getBackend() == Backend::CPU);
	REQUIRE(a == Tensor(2.718f, Backend::CPU));
}

#else

TEST_CASE("to() CUDA throws when CUDA not available", "[Tensor][to]") {
	Tensor a({3, 4}, 1.0f, Backend::CPU);

	REQUIRE_THROWS_AS(a.to(Backend::CUDA), std::runtime_error);
}

#endif
