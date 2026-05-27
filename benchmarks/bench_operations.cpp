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


static void BM_TensorAdd_ViewOffset_1000_1000(benchmark::State& state) {
	Tensor a({1000, 1000}, 1.0f, Backend::CPU);
	Tensor parent({2, 1000, 1000}, 2.0f, Backend::CPU);
	Tensor::View view = parent[1];
	for (auto _ : state) {
		auto result = a + view;
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(BM_TensorAdd_ViewOffset_1000_1000)->MinTime(2);


static void BM_TensorAdd_ViewStrided_1000_1000(benchmark::State& state) {
	Tensor a({1000, 1000}, 1.0f, Backend::CPU);
	Tensor parent({2000, 2000}, 2.0f, Backend::CPU);
	Tensor::View view = parent.subsample({2, 2});
	for (auto _ : state) {
		auto result = a + view;
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(BM_TensorAdd_ViewStrided_1000_1000)->MinTime(2);


static void BM_TensorAddInplace_1000_1000(benchmark::State& state) {
	Tensor a({1000, 1000}, 1.0f, Backend::CPU);
	Tensor b({1000, 1000}, 2.0f, Backend::CPU);
	for (auto _ : state) {
		a += b;
		benchmark::DoNotOptimize(a);
	}
}
BENCHMARK(BM_TensorAddInplace_1000_1000)->MinTime(2);


static void BM_TensorAddInplace_ViewOffset_1000_1000(benchmark::State& state) {
	Tensor parent({2, 1000, 1000}, 1.0f, Backend::CPU);
	Tensor::View view = parent[1];
	Tensor b({1000, 1000}, 2.0f, Backend::CPU);
	for (auto _ : state) {
		view += b;
		benchmark::DoNotOptimize(view);
	}
}
BENCHMARK(BM_TensorAddInplace_ViewOffset_1000_1000)->MinTime(2);


static void BM_TensorAddInplace_ViewStrided_1000_1000(benchmark::State& state) {
	Tensor a({1000, 1000}, 1.0f, Backend::CPU);
	Tensor parent({2000, 2000}, 2.0f, Backend::CPU);
	Tensor::View view = parent.subsample({2, 2});
	for (auto _ : state) {
		view += a;
		benchmark::DoNotOptimize(view);
	}
}
BENCHMARK(BM_TensorAddInplace_ViewStrided_1000_1000)->MinTime(2);


static void BM_TensorReduction_Sum_1000_1000(benchmark::State& state) {
	Tensor a({1000, 1000}, 1.0f, Backend::CPU);
	for (auto _ : state) {
		auto result = a.sum(1);
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(BM_TensorReduction_Sum_1000_1000)->MinTime(2);
