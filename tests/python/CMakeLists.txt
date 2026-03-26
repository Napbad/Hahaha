# Python Bindings Tests CMakeLists.txt
# Tests for the Python bindings

# Collect test source files using regex patterns
collect_test_sources("${CMAKE_CURRENT_SOURCE_DIR}" PYTHON_TEST_SOURCES)

# Check if Python bindings are being built
if (HAHAHA_BUILD_PYTHON AND PYTHON_TEST_SOURCES)
    # Create test executable for Python bindings
    add_executable(hahaha_test_python ${PYTHON_TEST_SOURCES})

    # Link against Python library and test framework
    target_link_libraries(hahaha_test_python
        PRIVATE
        hahaha::python
        GTest::gtest_main
    )

    # Apply architecture configuration
    apply_architecture_to_target(hahaha_test_python)

    # Include Google Test
    include(GoogleTest)
    gtest_discover_tests(hahaha_test_python)
endif ()
