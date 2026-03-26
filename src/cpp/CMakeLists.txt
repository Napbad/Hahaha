# C++ Interface Library CMakeLists.txt
# ML library C++ interface layer

# Collect source files using regex patterns
collect_cpp_sources("${CMAKE_CURRENT_SOURCE_DIR}" CPP_CPP_SOURCES)
collect_cuda_sources("${CMAKE_CURRENT_SOURCE_DIR}" CPP_CUDA_SOURCES)

# Create C++ interface library target
if (HAHAHA_USE_CUDA)
    add_library(hahaha_cpp STATIC ${CPP_CPP_SOURCES} ${CPP_CUDA_SOURCES})
else ()
    add_library(hahaha_cpp STATIC ${CPP_CPP_SOURCES})
endif ()

add_library(hahaha::cpp ALIAS hahaha_cpp)

# Set include directories
target_include_directories(hahaha_cpp
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/cpp/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link against core library
target_link_libraries(hahaha_cpp
    PUBLIC
    hahaha::core
)

# Apply architecture configuration
apply_architecture_to_target(hahaha_cpp)

# Configure x86 SIMD if applicable
configure_x86_simd(hahaha_cpp)
