# Core Tests CMakeLists.txt
# Tests for the core ML library

# Collect test source files using regex patterns
collect_test_sources("${CMAKE_CURRENT_SOURCE_DIR}" CORE_TEST_SOURCES)

# Create test executable if there are test files
if (CORE_TEST_SOURCES)
    add_executable(hahaha_test_core ${CORE_TEST_SOURCES})

    # Link against core library and test framework
    target_link_libraries(hahaha_test_core
        PRIVATE
        hahaha::core
        GTest::gtest_main
    )

    # Apply architecture configuration
    apply_architecture_to_target(hahaha_test_core)

    # Include Google Test
    include(GoogleTest)
    gtest_discover_tests(hahaha_test_core)
endif ()
