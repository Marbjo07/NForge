#include <benchmark/benchmark.h>

#include "nforge/core/tensor_layout.h"

static void BM_PhysicalOffset_2D_1000(benchmark::State& state) {
	size_t N = 1000ul;
	TensorLayout layout = Tensor::Shape({N, N}).toContiguousLayout();
	size_t count = N * N;

	for (auto _ : state) {
		size_t sum = 0;
		for (size_t i = 0; i < count; i++) {
			sum += physicalOffset(i, layout);
		}
		benchmark::DoNotOptimize(sum);
	}
}
BENCHMARK(BM_PhysicalOffset_2D_1000)->MinTime(2);


static void BM_PhysicalOffset_3D_100(benchmark::State& state) {
	size_t N = 100ul;
	TensorLayout layout = Tensor::Shape({N, N, N}).toContiguousLayout();
	size_t count = N * N * N;

	for (auto _ : state) {
		size_t sum = 0;
		for (size_t i = 0; i < count; i++) {
			sum += physicalOffset(i, layout);
		}
		benchmark::DoNotOptimize(sum);
	}
}
BENCHMARK(BM_PhysicalOffset_3D_100)->MinTime(2);


static void BM_PhysicalOffset_Broadcast_1000(benchmark::State& state) {
	size_t N = 1000ul;
	// broadcast layout: shape=(N,N), strides=(0,1), first dim is broadcast
	TensorLayout layout(std::array<size_t, MAX_DIMS>{N, N},
	                    std::array<size_t, MAX_DIMS>{size_t(0), size_t(1)}, size_t(0), size_t(2));
	size_t count = N * N;

	for (auto _ : state) {
		size_t sum = 0;
		for (size_t i = 0; i < count; i++) {
			sum += physicalOffset(i, layout);
		}
		benchmark::DoNotOptimize(sum);
	}
}
BENCHMARK(BM_PhysicalOffset_Broadcast_1000)->MinTime(2);


static void BM_PhysicalOffset_Strided_1000(benchmark::State& state) {
	size_t N = 1000ul;
	// non-contiguous strided layout, strides=(2000,2)
	TensorLayout layout(std::array<size_t, MAX_DIMS>{N, N},
	                    std::array<size_t, MAX_DIMS>{size_t(2000), size_t(2)}, size_t(0),
	                    size_t(2));
	size_t count = N * N;

	for (auto _ : state) {
		size_t sum = 0;
		for (size_t i = 0; i < count; i++) {
			sum += physicalOffset(i, layout);
		}
		benchmark::DoNotOptimize(sum);
	}
}
BENCHMARK(BM_PhysicalOffset_Strided_1000)->MinTime(2);
