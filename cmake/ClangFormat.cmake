find_program(CLANG_FORMAT_EXE NAMES clang-format)

if(CLANG_FORMAT_EXE)
    file(GLOB_RECURSE ALL_SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/src/*.cu
        ${CMAKE_SOURCE_DIR}/src/*.h
        ${CMAKE_SOURCE_DIR}/src/*.cuh

        ${CMAKE_SOURCE_DIR}/scenarios/*.h
        
        ${CMAKE_SOURCE_DIR}/include/*.h
        
        ${CMAKE_SOURCE_DIR}/tests/*.h
        ${CMAKE_SOURCE_DIR}/tests/*.cpp
    )

    if(NOT ALL_SOURCE_FILES)
        message(WARNING "NForge: clang-format found no source files to format")
    else()
        message(STATUS "NForge: clang-format will process ${ALL_SOURCE_FILES}")
    endif()

    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXE}
            --style=file:${CMAKE_SOURCE_DIR}/.clang-format
            -i ${ALL_SOURCE_FILES}
        COMMENT "Formatting source files..."
    )

    add_custom_target(format-check
        COMMAND ${CLANG_FORMAT_EXE}
            --style=file:${CMAKE_SOURCE_DIR}/.clang-format
            --dry-run --Werror ${ALL_SOURCE_FILES}
        COMMENT "Checking formatting..."
    )
endif()