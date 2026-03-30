# Core ML Library CMakeLists.txt
# This is the main part of the ML library containing core implementations

# Collect source files using regex patterns
collect_cpp_sources("${CMAKE_CURRENT_SOURCE_DIR}" CORE_CPP_SOURCES)
collect_cuda_sources("${CMAKE_CURRENT_SOURCE_DIR}" CORE_CUDA_SOURCES)

# Create core library target
if (HAHAHA_USE_CUDA)
    add_library(hahaha_core STATIC ${CORE_CPP_SOURCES} ${CORE_CUDA_SOURCES})
else ()
    add_library(hahaha_core STATIC ${CORE_CPP_SOURCES})
endif ()

include_directories(${PROJECT_SOURCE_DIR}/src/core/include)

add_library(hahaha::core ALIAS hahaha_core)

# Set include directories
target_include_directories(hahaha_core
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Apply architecture configuration
apply_architecture_to_target(hahaha_core)

# Configure x86 SIMD if applicable
configure_x86_simd(hahaha_core)

# Link system libraries
link_system_libraries(hahaha_core)

# Link CUDA if enabled
link_cuda_libraries(hahaha_core)
