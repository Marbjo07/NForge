file(GLOB PHYSICS_SOURCES "*.cpp")
add_nforge_test_binary(test_nforge_physics ${PHYSICS_SOURCES})

target_include_directories(test_nforge_physics PRIVATE ${CMAKE_SOURCE_DIR}/scenarios)