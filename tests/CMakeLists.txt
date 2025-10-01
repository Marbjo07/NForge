cmake_minimum_required(VERSION 3.14)
project(NForgeTests)

include(FetchContent)
FetchContent_Declare(
	Catch2
	GIT_REPOSITORY https://github.com/catchorg/Catch2.git
	GIT_TAG        v3.8.1
)
FetchContent_MakeAvailable(Catch2)

# Build test binary
add_executable(test_nforge test_tensor.cpp)
target_link_libraries(test_nforge PRIVATE NForge Catch2::Catch2WithMain)

# Enable testing and add Catch2 tests to CTest
include(CTest)
include(Catch)
catch_discover_tests(test_nforge)
