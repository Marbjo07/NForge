#include <benchmark/benchmark.h>

#include "physics/projectile_motion.h"
#include "physics/sphere_slide.h"

static void BM_Physics_ProjectileMotion(benchmark::State& state) {
	ProjectileMotionParams params;
	for (auto _ : state) {
		benchmark::DoNotOptimize(simulateProjectileMotion(params));
	}
}
BENCHMARK(BM_Physics_ProjectileMotion)->MinTime(2);


static void BM_Physics_SphereSlide(benchmark::State& state) {
	SphereSlideParams params;
	for (auto _ : state) {
		benchmark::DoNotOptimize(simulateSphereSlide(params));
	}
}
BENCHMARK(BM_Physics_SphereSlide)->MinTime(2);
