# Helpers to add a CXX flag guarded by compiler, config, and language
function(nforge_cxx_flag target visibility config flag)
    target_compile_options(${target} ${visibility}
        $<$<AND:$<CONFIG:${config}>,$<COMPILE_LANGUAGE:CXX>>:${flag}>)
endfunction()

function(nforge_cuda_flag target visibility config flag)
    target_compile_options(${target} ${visibility}
        $<$<AND:$<CONFIG:${config}>,$<COMPILE_LANGUAGE:CUDA>>:${flag}>)
endfunction()

function(nforge_link_flag target visibility config flag)
    target_link_options(${target} ${visibility}
        $<$<CONFIG:${config}>:${flag}>)
endfunction()


function(nforge_apply_compiler_flags target)
    ## Release: CXX 
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        nforge_cxx_flag(${target} PRIVATE Release -march=native)
        nforge_cxx_flag(${target} PRIVATE Release -ffast-math)
        nforge_cxx_flag(${target} PRIVATE Release -funroll-loops)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        nforge_cxx_flag(${target} PRIVATE Release /arch:AVX2)
        nforge_cxx_flag(${target} PRIVATE Release /fp:fast)
    else()
        message(WARNING "NForge: unrecognised CXX compiler '${CMAKE_CXX_COMPILER_ID}', no release flags set")
    endif()

    ## Debug: CXX and linker
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        nforge_cxx_flag (${target} PUBLIC Debug -fsanitize=address)
        nforge_cxx_flag (${target} PUBLIC Debug -fsanitize=undefined)
        nforge_link_flag(${target} PUBLIC Debug -fsanitize=address)
        nforge_link_flag(${target} PUBLIC Debug -fsanitize=undefined)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        nforge_cxx_flag(${target} PUBLIC Debug /fsanitize=address)
    endif()

    ## CUDA 
    if(NFORGE_ENABLE_CUDA)
        nforge_cuda_flag(${target} PRIVATE Release --use_fast_math)
        nforge_cuda_flag(${target} PRIVATE Debug   -G)

        # allow constexpr host functions in device functions
        nforge_cuda_flag(${target} PRIVATE Release --expt-relaxed-constexpr)
        nforge_cuda_flag(${target} PRIVATE Debug --expt-relaxed-constexpr)
    endif()
endfunction()