include(FetchContent)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.1
    GIT_SHALLOW    TRUE
)

# Remove GTest depend
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)

FetchContent_MakeAvailable(googlebenchmark)

file(GLOB BENCHMARK_SOURCES "*.cpp")
add_executable(nforge_benchmarks ${BENCHMARK_SOURCES})

target_include_directories(nforge_benchmarks PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/scenarios
)

target_link_libraries(nforge_benchmarks PRIVATE
    NForge
    benchmark::benchmark_main
)