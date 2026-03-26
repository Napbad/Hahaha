# Tests Directory CMakeLists.txt
# Aggregates all test subdirectories

# Enable testing
enable_testing()

# Include core tests
add_subdirectory(core)

# Include C++ interface tests
add_subdirectory(cpp)

# Include Python tests (optional)
add_subdirectory(python)
