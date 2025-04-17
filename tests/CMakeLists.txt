# Pull Catch2
include(FetchContent)
FetchContent_Declare(
	Catch2
	GIT_REPOSITORY https://github.com/catchorg/Catch2.git
	GIT_TAG        v3.8.1
)
FetchContent_MakeAvailable(Catch2)

# Build test binary
add_executable(test_nforge test_tensor.cpp)
target_link_libraries(test_nforge PRIVATE nforge Catch2::Catch2WithMain)

# Add Catch2 tests to CTest
include(CTest)
include(Catch)
catch_discover_tests(test_nforge)
