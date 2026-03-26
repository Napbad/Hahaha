# Python Bindings CMakeLists.txt
# Python interface for the ML library using pybind11

# Collect source files using regex patterns
collect_python_sources("${CMAKE_CURRENT_SOURCE_DIR}" PYTHON_SOURCES)

# Check if Python bindings should be built
option(HAHAHA_BUILD_PYTHON "Build Python bindings" OFF)

if (HAHAHA_BUILD_PYTHON)
    # Find Python and pybind11
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        # Try without CONFIG if not found via vcpkg
        find_package(pybind11 REQUIRED)
    endif()

    # Create Python module
    pybind11_add_module(hahaha_python ${PYTHON_SOURCES})

    add_library(hahaha::python ALIAS hahaha_python)

    # Set include directories
    target_include_directories(hahaha_python
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/python/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    # Link against core library
    target_link_libraries(hahaha_python
        PUBLIC
        hahaha::core
        pybind11::module
    )

    # Apply architecture configuration
    apply_architecture_to_target(hahaha_python)

    # Configure installation paths
    install(TARGETS hahaha_python
        LIBRARY DESTINATION ${Python_SITEARCH}
    )
endif ()
