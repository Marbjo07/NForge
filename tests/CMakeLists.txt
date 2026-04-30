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

include(CTest)
include(Catch)

# Helper used by all subdirectories
function(add_nforge_test_binary target)
    # ARGN is the source files passed after target name
    add_executable(${target} ${ARGN})
	target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/src)
	target_link_libraries(${target} PRIVATE NForge Catch2::Catch2WithMain)

	if(DEFINED NFORGE_ENABLE_CUDA AND NFORGE_ENABLE_CUDA)
		find_package(CUDAToolkit REQUIRED)
		target_link_libraries(${target} PRIVATE CUDA::cudart)
	endif()

    catch_discover_tests(${target})
endfunction()

add_subdirectory(unit)
add_subdirectory(integration)