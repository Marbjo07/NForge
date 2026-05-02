#include <benchmark/benchmark.h>
#include "physics/projectile_motion.h"
#include "physics/sphere_slide.h"

static void BM_ProjectileMotion(benchmark::State& state) {
    ProjectileMotionParams params;
    for (auto _ : state) {
        benchmark::DoNotOptimize(simulateProjectileMotion(params));
    }
}

static void BM_SphereSlide(benchmark::State& state) {
    SphereSlideParams params;
    for (auto _ : state) {
        benchmark::DoNotOptimize(simulateSphereSlide(params));
    }
}


BENCHMARK(BM_ProjectileMotion)->Iterations(1000);
BENCHMARK(BM_SphereSlide)->Iterations(100);