# Example Directory CMakeLists.txt
# Aggregates all example subdirectories

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/core/include)

# Include core examples
add_subdirectory(core)
