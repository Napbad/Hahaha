# Tests Directory CMakeLists.txt
# Aggregates all test subdirectories

# Enable testing
enable_testing()

include_directories(${CMAKE_SOURCE_DIR}/src/core/include)

# Include core tests
add_subdirectory(core)
