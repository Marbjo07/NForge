#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "nforge/nforge.h"
#include "utils.h"

TEST_CASE("transfer preserves data", "[Tensor][transfer]") {
	auto srcBackend = GENERATE(from_range(backends));
	auto tgtBackend = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(srcBackend) + " -> " + getBackendString(tgtBackend)) {
		Tensor src({8, 4, 5}, srcBackend);
		src.fillRand();

		auto original = src.toVector();

		src.to(tgtBackend);

		REQUIRE(src.getBackend() == tgtBackend);
		REQUIRE(src.toVector() == original);
	}
}

TEST_CASE("views update after transfer", "[View][transfer]") {
	auto src = GENERATE(from_range(backends));
	auto tgt = GENERATE(from_range(backends));

	DYNAMIC_SECTION(getBackendString(src) + " -> " + getBackendString(tgt)) {
		Tensor parent({4, 4}, src);
		parent.fillRand();

		auto row0 = parent[0];
		auto row1 = parent[1];
		auto sub = parent.subsample({2, 2});

		auto r0_orig = row0.toVector();
		auto r1_orig = row1.toVector();
		auto sub_orig = sub.toVector();

		parent.to(tgt);

		REQUIRE(row0.getBackend() == tgt);
		REQUIRE(row1.getBackend() == tgt);
		REQUIRE(sub.getBackend() == tgt);

		REQUIRE(row0.toVector() == r0_orig);
		REQUIRE(row1.toVector() == r1_orig);
		REQUIRE(sub.toVector() == sub_orig);
	}
}

#ifndef NFORGE_WITH_CUDA

TEST_CASE("transfer CUDA throws when CUDA not available", "[Tensor][transfer]") {
	Tensor a({3, 4}, 1.0f, Backend::CPU);

	REQUIRE_THROWS_AS(a.to(Backend::CUDA), std::runtime_error);
}

#endif
