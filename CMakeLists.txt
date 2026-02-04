cmake_minimum_required(VERSION 3.16)
project(NForge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output dirs
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Sources
set(NFORGE_SRC
	src/core/tensor.cpp
	src/core/tensor_view.cpp
	src/core/tensor_shape.cpp
	src/backend/cpu/tensor_impl_CPU.cpp
)

# Build as a dynamic library
add_library(NForge SHARED ${NFORGE_SRC})

# Include headers
target_include_directories(NForge PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Enable testing
enable_testing()
add_subdirectory(tests)