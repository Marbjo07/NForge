cmake_minimum_required(VERSION 3.16)
project(NForge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_WITH_CUDA "Build with CUDA support" OFF)


# Output dirs
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Sources
set(NFORGE_SRC
	src/tensor.cpp
	src/tensor_view.cpp
	src/tensor_impl_CPU.cpp
)

if(BUILD_WITH_CUDA)
	enable_language(CUDA)
	list(APPEND NFORGE_SRC src/tensor_impl_CUDA.cu)
endif()

# Build as a dynamic library
add_library(NForge SHARED ${NFORGE_SRC})

# Include headers
target_include_directories(NForge PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(WIN32)
    set_target_properties(NForge PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(BUILD_WITH_CUDA)
	target_compile_definitions(NForge PRIVATE NFORGE_USE_CUDA)
endif()

# Enable testing
enable_testing()
add_subdirectory(tests)