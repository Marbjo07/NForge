cmake_minimum_required(VERSION 3.16)
project(NForgeTests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
	Catch2
	GIT_REPOSITORY https://github.com/catchorg/Catch2.git
	GIT_TAG        v3.8.1
)
FetchContent_MakeAvailable(Catch2)


# Add all test files
file(GLOB TEST_SOURCES "*.cpp")


# Build test binary
add_executable(test_nforge ${TEST_SOURCES})
target_include_directories(test_nforge PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_nforge PRIVATE NForge Catch2::Catch2WithMain)


# Enable testing and add Catch2 tests to CTest
include(CTest)
include(Catch)
catch_discover_tests(test_nforge)