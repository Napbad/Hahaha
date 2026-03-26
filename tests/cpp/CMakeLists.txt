# C++ Interface Tests CMakeLists.txt
# Tests for the C++ interface layer

# Collect test source files using regex patterns
collect_test_sources("${CMAKE_CURRENT_SOURCE_DIR}" CPP_TEST_SOURCES)

# Create test executable if there are test files
if (CPP_TEST_SOURCES)
    add_executable(hahaha_test_cpp ${CPP_TEST_SOURCES})

    # Link against C++ library and test framework
    target_link_libraries(hahaha_test_cpp
        PRIVATE
        hahaha::cpp
        GTest::gtest_main
    )

    # Apply architecture configuration
    apply_architecture_to_target(hahaha_test_cpp)

    # Include Google Test
    include(GoogleTest)
    gtest_discover_tests(hahaha_test_cpp)
endif ()
