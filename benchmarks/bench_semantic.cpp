#include <benchmark/benchmark.h>

#include "nforge/nforge.h"
#include "ops/semantic/semantic.h"

static void BM_BinaryOpContext_SameShape_1000(benchmark::State& state) {
	Tensor a({1000, 1000}, 1.0f, Backend::CPU);
	Tensor b({1000, 1000}, 2.0f, Backend::CPU);
	for (auto _ : state) {
		auto ctx = semantic::BinaryOpContext::build(a, b);
		benchmark::DoNotOptimize(ctx);
	}
}
BENCHMARK(BM_BinaryOpContext_SameShape_1000)->MinTime(2);


static void BM_BinaryOpContext_Broadcast(benchmark::State& state) {
	Tensor a({3, 1}, 1.0f, Backend::CPU);
	Tensor b({1, 4}, 2.0f, Backend::CPU);
	for (auto _ : state) {
		auto ctx = semantic::BinaryOpContext::build(a, b);
		benchmark::DoNotOptimize(ctx);
	}
}
BENCHMARK(BM_BinaryOpContext_Broadcast)->MinTime(2);


static void BM_BinaryOpContext_Scalar_1000(benchmark::State& state) {
	Tensor a({1}, 1.0f, Backend::CPU);
	Tensor b({1000, 1000}, 2.0f, Backend::CPU);
	for (auto _ : state) {
		auto ctx = semantic::BinaryOpContext::build(a, b);
		benchmark::DoNotOptimize(ctx);
	}
}
BENCHMARK(BM_BinaryOpContext_Scalar_1000)->MinTime(2);


static void BM_MatmulContext_2D(benchmark::State& state) {
	Tensor a({100, 200}, 1.0f, Backend::CPU);
	Tensor b({200, 300}, 2.0f, Backend::CPU);
	for (auto _ : state) {
		auto ctx = semantic::MatmulContext::build(a, b);
		benchmark::DoNotOptimize(ctx);
	}
}
BENCHMARK(BM_MatmulContext_2D)->MinTime(2);


static void BM_MatmulContext_3D_Batched(benchmark::State& state) {
	Tensor a({5, 100, 200}, 1.0f, Backend::CPU);
	Tensor b({5, 200, 300}, 2.0f, Backend::CPU);
	for (auto _ : state) {
		auto ctx = semantic::MatmulContext::build(a, b);
		benchmark::DoNotOptimize(ctx);
	}
}
BENCHMARK(BM_MatmulContext_3D_Batched)->MinTime(2);


static void BM_ReductionContext_1000(benchmark::State& state) {
	Tensor a({1000, 1000}, 1.0f, Backend::CPU);
	for (auto _ : state) {
		auto ctx = semantic::ReductionContext::build(a, 1);
		benchmark::DoNotOptimize(ctx);
	}
}
BENCHMARK(BM_ReductionContext_1000)->MinTime(2);


static void BM_IndexContext_1000(benchmark::State& state) {
	Tensor a({1000, 1000}, 1.0f, Backend::CPU);
	for (auto _ : state) {
		auto ctx = semantic::IndexContext::build(a, 0);
		benchmark::DoNotOptimize(ctx);
	}
}
BENCHMARK(BM_IndexContext_1000)->MinTime(2);
