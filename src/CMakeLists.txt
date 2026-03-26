# Source Directory CMakeLists.txt
# Aggregates all source subdirectories

# Include core ML library
add_subdirectory(core)

# Include C++ interface layer
add_subdirectory(cpp)

# Include Python bindings (optional)
add_subdirectory(python)
