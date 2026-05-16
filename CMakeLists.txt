cmake_minimum_required(VERSION 3.16)
project(NForge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NFORGE_ENABLE_CUDA "Enable CUDA support" OFF)
option(NFORGE_BUILD_BENCHMARKS "Build benchmarks" ON)


## Sources
set(NFORGE_SRC
    src/core/tensor.cpp
    src/core/tensor_view.cpp
    src/core/tensor_shape.cpp
    src/core/tensor_layout.cpp
    src/backend/cpu/tensor_impl_CPU.cpp
    src/ops/semantic/semantic.cpp
)

## Add CUDA sources
if(NFORGE_ENABLE_CUDA)
    enable_language(CUDA)

    list(APPEND NFORGE_SRC
        src/backend/cuda/kernels/kernels.cu
        src/backend/cuda/tensor_impl_CUDA.cu
    )

    add_compile_definitions(NFORGE_WITH_CUDA)

    find_package(CUDAToolkit QUIET)
endif()

add_library(NForge STATIC ${NFORGE_SRC})


## Includes
target_include_directories(NForge
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)


## Compiler flags for configs
include(cmake/CompilerFlags.cmake)
nforge_apply_compiler_flags(NForge)


## Configure CUDA dependencies
if(NFORGE_ENABLE_CUDA)
    target_link_libraries(NForge PUBLIC CUDA::cudart CUDA::curand)
    
    set_property(TARGET NForge PROPERTY CUDA_ARCHITECTURES native)
endif()


## Testing
enable_testing()
add_subdirectory(tests)


## Benchmarks
if(NFORGE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

## Formatting
include(cmake/ClangFormat.cmake)