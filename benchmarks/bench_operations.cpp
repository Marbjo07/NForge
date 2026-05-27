#include <benchmark/benchmark.h>

#include "nforge/nforge.h"

static void BM_TensorAdd_1000_1000(benchmark::State& state) {
	Tensor a({1000, 1000}, 1.0f, Backend::CPU);
	Tensor b({1000, 1000}, 2.0f, Backend::CPU);
	for (auto _ : state) {
		auto result = a + b;
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(BM_TensorAdd_1000_1000)->MinTime(2);

static void BM_TensorAdd_Broadcast_1000(benchmark::State& state) {
	Tensor a({1000, 1}, 1.0f, Backend::CPU);
	Tensor b({1, 1000}, 2.0f, Backend::CPU);
	for (auto _ : state) {
		auto result = a + b;
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(BM_TensorAdd_Broadcast_1000)->MinTime(2);

static void BM_TensorAddInplace_1000_1000(benchmark::State& state) {
	Tensor a({1000, 1000}, 1.0f, Backend::CPU);
	Tensor b({1000, 1000}, 2.0f, Backend::CPU);
	for (auto _ : state) {
		a += b;
		benchmark::DoNotOptimize(a);
	}
}
BENCHMARK(BM_TensorAddInplace_1000_1000)->MinTime(2);

static void BM_TensorMatmul_32_32(benchmark::State& state) {
	Tensor a({32, 32}, 1.0f, Backend::CPU);
	Tensor b({32, 32}, 2.0f, Backend::CPU);
	for (auto _ : state) {
		auto result = a.matmul(b);
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(BM_TensorMatmul_32_32)->MinTime(2);

static void BM_TensorMatmul_256_256(benchmark::State& state) {
	Tensor a({256, 256}, 1.0f, Backend::CPU);
	Tensor b({256, 256}, 2.0f, Backend::CPU);
	for (auto _ : state) {
		auto result = a.matmul(b);
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(BM_TensorMatmul_256_256)->MinTime(2);

static void BM_TensorMatmul_512_512(benchmark::State& state) {
	Tensor a({512, 512}, 1.0f, Backend::CPU);
	Tensor b({512, 512}, 2.0f, Backend::CPU);
	for (auto _ : state) {
		auto result = a.matmul(b);
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(BM_TensorMatmul_512_512)->MinTime(2);

static void BM_TensorReduction_Sum_1000_1000(benchmark::State& state) {
	Tensor a({1000, 1000}, 1.0f, Backend::CPU);
	for (auto _ : state) {
		auto result = a.sum(1);
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(BM_TensorReduction_Sum_1000_1000)->MinTime(2);
