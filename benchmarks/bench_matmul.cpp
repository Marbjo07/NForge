#include <benchmark/benchmark.h>

#include "nforge/nforge.h"

static void BM_TensorMatmul_32_32(benchmark::State& state) {
	Tensor a({32, 32}, 1.0f, Backend::CPU);
	Tensor b({32, 32}, 2.0f, Backend::CPU);
	for (auto _ : state) {
		auto result = a.matmul(b);
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(BM_TensorMatmul_32_32)->MinTime(2.0);


static void BM_TensorMatmul_256_256(benchmark::State& state) {
	Tensor a({256, 256}, 1.0f, Backend::CPU);
	Tensor b({256, 256}, 2.0f, Backend::CPU);
	for (auto _ : state) {
		auto result = a.matmul(b);
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(BM_TensorMatmul_256_256)->MinTime(2.0);


static void BM_TensorMatmul_512_512(benchmark::State& state) {
	Tensor a({512, 512}, 1.0f, Backend::CPU);
	Tensor b({512, 512}, 2.0f, Backend::CPU);
	for (auto _ : state) {
		auto result = a.matmul(b);
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(BM_TensorMatmul_512_512)->MinTime(2.0);


static void BM_TensorMatmul_ViewOffset_256_256(benchmark::State& state) {
	Tensor a({256, 256}, 1.0f, Backend::CPU);
	Tensor parent({2, 256, 256}, 2.0f, Backend::CPU);
	Tensor::View view = parent[1];
	for (auto _ : state) {
		auto result = a.matmul(view);
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(BM_TensorMatmul_ViewOffset_256_256)->MinTime(2.0);


static void BM_TensorMatmul_ViewStrided_256_256(benchmark::State& state) {
	Tensor a({256, 256}, 1.0f, Backend::CPU);
	Tensor parent({512, 512}, 2.0f, Backend::CPU);
	Tensor::View view = parent.subsample({2, 2});
	for (auto _ : state) {
		auto result = a.matmul(view);
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(BM_TensorMatmul_ViewStrided_256_256)->MinTime(2.0);
