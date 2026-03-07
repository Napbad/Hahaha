cmake_minimum_required(VERSION 3.20)
if (MSVC)
    add_compile_options("/execution_charset:utf-8" "/source_charset:utf-8")
    set(ENV{VSLANG} "1033")
endif ()

# ---- vcpkg toolchain detection ----
# Prefer repo-local vcpkg (created by ./dev/dev_env_setup.py), but allow users to
# pass -DCMAKE_TOOLCHAIN_FILE=... or set VCPKG_ROOT.
#
# Note: This must run before project()/enable_language() so the toolchain can
# affect compiler detection.
if ((NOT DEFINED CMAKE_TOOLCHAIN_FILE OR CMAKE_TOOLCHAIN_FILE STREQUAL "") OR (DEFINED CMAKE_TOOLCHAIN_FILE AND NOT EXISTS "${CMAKE_TOOLCHAIN_FILE}"))
    if (DEFINED CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE AND NOT EXISTS "${CMAKE_TOOLCHAIN_FILE}")
        message(WARNING "CMAKE_TOOLCHAIN_FILE does not exist: '${CMAKE_TOOLCHAIN_FILE}'")
    endif ()

    set(_repo_vcpkg_toolchain "${CMAKE_CURRENT_LIST_DIR}/vcpkg/vcpkg_root/scripts/buildsystems/vcpkg.cmake")
    if (EXISTS "${_repo_vcpkg_toolchain}")
        set(CMAKE_TOOLCHAIN_FILE "${_repo_vcpkg_toolchain}" CACHE FILEPATH "vcpkg toolchain file" FORCE)
        message(STATUS "Using repo-local vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
    elseif (DEFINED ENV{VCPKG_ROOT} AND EXISTS "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "vcpkg toolchain file" FORCE)
        message(STATUS "Using vcpkg toolchain from VCPKG_ROOT: ${CMAKE_TOOLCHAIN_FILE}")
    else ()
        message(FATAL_ERROR
                "vcpkg toolchain not set or not found. Please either:\n"
                "  1) Run ./dev/dev_env_setup.py (creates <repo>/vcpkg/vcpkg_root)\n"
                "  2) Set VCPKG_ROOT to your vcpkg root (e.g. <repo>/vcpkg/vcpkg_root)\n"
                "  3) Pass the toolchain: -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake"
        )
    endif ()
endif ()

# Use static triplet on Windows so vcpkg builds GTest (and deps) with static CRT.
# This avoids heap corruption (debug_heap _acrt_first_block == header) when the exe
# uses /MTd and GTest would otherwise be a DLL with /MDd (separate heaps).
if (WIN32 AND (NOT DEFINED VCPKG_TARGET_TRIPLET OR VCPKG_TARGET_TRIPLET STREQUAL ""))
    set(VCPKG_TARGET_TRIPLET "x64-windows-static" CACHE STRING "vcpkg triplet (static CRT to match this project)")
endif ()

#=========================== Project Configuration ===================================
project(hahaha
        VERSION 0.0.1
        LANGUAGES CXX
)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MSVC: Use static runtime library to match vcpkg x64-windows-static triplet
# This avoids LNK2038 "RuntimeLibrary mismatch" errors
if (MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif ()

#=========================== Architecture Detection ===================================
# Detect architecture and define macros (fallback to macros.h if not detected here).
# When cross-compiling, CMAKE_SYSTEM_PROCESSOR is the *target* arch; you can also
# explicitly set HAHAHA_TARGET_ARCH (e.g. "aarch64", "x86_64") to override.
set(HAHAHA_TARGET_ARCH "" CACHE STRING
        "Override target architecture (e.g. aarch64, x86_64). Empty = use CMAKE_SYSTEM_PROCESSOR (normal for cross-compile).")

set(_arch_to_use "${CMAKE_SYSTEM_PROCESSOR}")
if (HAHAHA_TARGET_ARCH)
    set(_arch_to_use "${HAHAHA_TARGET_ARCH}")
endif ()

# Determine bitness (target pointer size when cross-compiling)
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(HAHAHA_ARCH_BITS 64)
elseif (CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(HAHAHA_ARCH_BITS 32)
else ()
    set(HAHAHA_ARCH_BITS 0)
endif ()

# Detect architecture from target processor (or override)
string(TOLOWER "${_arch_to_use}" _processor_lower)

if (_processor_lower MATCHES "^(amd64|x86_64|x64)$")
    set(HAHAHA_ARCH_X86_64 1)
    set(HAHAHA_ARCH_IS_X86_FAMILY 1)
    set(HAHAHA_ARCH_NAME "x86-64")
elseif (_processor_lower MATCHES "^(i[3-6]86|x86|i86pc)$")
    set(HAHAHA_ARCH_X86_32 1)
    set(HAHAHA_ARCH_IS_X86_FAMILY 1)
    set(HAHAHA_ARCH_NAME "x86")
elseif (_processor_lower MATCHES "^(aarch64|arm64|armv8)$")
    set(HAHAHA_ARCH_ARM64 1)
    set(HAHAHA_ARCH_NAME "arm64")
elseif (_processor_lower MATCHES "^(arm|armv7|armv6|armv5)$")
    set(HAHAHA_ARCH_ARM32 1)
    set(HAHAHA_ARCH_NAME "arm32")
elseif (_processor_lower MATCHES "^(ppc64|powerpc64)$")
    set(HAHAHA_ARCH_PPC64 1)
    set(HAHAHA_ARCH_NAME "ppc64")
elseif (_processor_lower MATCHES "^(riscv64|rv64)$")
    set(HAHAHA_ARCH_RISCV64 1)
    set(HAHAHA_ARCH_NAME "riscv64")
else ()
    message(WARNING "Unknown architecture: ${_arch_to_use}")
    set(HAHAHA_ARCH_UNKNOWN 1)
    set(HAHAHA_ARCH_NAME "unknown")
endif ()

# Architecture macros will be added to target after it's created (see below)

#=========================== Dependencies Management ===================================
# ---- core library ----
# Full configuration: include all source files
file(GLOB_RECURSE HAHAHA_CPP_SOURCES CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/core/src/*.cpp"
)

file(
        GLOB_RECURSE HAHAHA_CUDA_SOURCES CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/core/src/*.cu"
)

include_directories("${PROJECT_SOURCE_DIR}/core/include")

# Enable CUDA language before any target uses .cu sources (required for CMAKE_CUDA_COMPILE_OBJECT etc.)
if (HAHAHA_USE_CUDA)
    if (MSVC)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    endif ()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

if (HAHAHA_USE_CUDA)
    add_library(hahaha STATIC ${HAHAHA_CPP_SOURCES} ${HAHAHA_CUDA_SOURCES})
    add_compile_definitions(HAHAHA_USE_CUDA)
else ()
    add_library(hahaha STATIC ${HAHAHA_CPP_SOURCES})
endif ()
add_library(hahaha::hahaha ALIAS hahaha)

target_include_directories(hahaha
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Define architecture macros via compile definitions
# Note: macros.h will check #ifndef and skip redefinition, so it serves as fallback
if (HAHAHA_ARCH_X86_64)
    target_compile_definitions(hahaha PUBLIC
            HAHAHA_ARCH_X86_64=1
            HAHAHA_ARCH_IS_X86_FAMILY=1
            HAHAHA_ARCH_BITS=${HAHAHA_ARCH_BITS}
            HAHAHA_ARCH_NAME="${HAHAHA_ARCH_NAME}"
    )
elseif (HAHAHA_ARCH_X86_32)
    target_compile_definitions(hahaha PUBLIC
            HAHAHA_ARCH_X86_32=1
            HAHAHA_ARCH_IS_X86_FAMILY=1
            HAHAHA_ARCH_BITS=${HAHAHA_ARCH_BITS}
            HAHAHA_ARCH_NAME="${HAHAHA_ARCH_NAME}"
    )
elseif (HAHAHA_ARCH_ARM64)
    target_compile_definitions(hahaha PUBLIC
            HAHAHA_ARCH_ARM64=1
            HAHAHA_ARCH_BITS=${HAHAHA_ARCH_BITS}
            HAHAHA_ARCH_NAME="${HAHAHA_ARCH_NAME}"
    )
elseif (HAHAHA_ARCH_ARM32)
    target_compile_definitions(hahaha PUBLIC
            HAHAHA_ARCH_ARM32=1
            HAHAHA_ARCH_BITS=${HAHAHA_ARCH_BITS}
            HAHAHA_ARCH_NAME="${HAHAHA_ARCH_NAME}"
    )
elseif (HAHAHA_ARCH_PPC64)
    target_compile_definitions(hahaha PUBLIC
            HAHAHA_ARCH_PPC64=1
            HAHAHA_ARCH_BITS=${HAHAHA_ARCH_BITS}
            HAHAHA_ARCH_NAME="${HAHAHA_ARCH_NAME}"
    )
elseif (HAHAHA_ARCH_RISCV64)
    target_compile_definitions(hahaha PUBLIC
            HAHAHA_ARCH_RISCV64=1
            HAHAHA_ARCH_BITS=${HAHAHA_ARCH_BITS}
            HAHAHA_ARCH_NAME="${HAHAHA_ARCH_NAME}"
    )
elseif (HAHAHA_ARCH_UNKNOWN)
    target_compile_definitions(hahaha PUBLIC HAHAHA_ARCH_UNKNOWN=1)
endif ()

# ---- x86 SIMD (optional): SSE2 is baseline on x64; enable AVX/AVX2 for 256-bit and integer ----
if (HAHAHA_ARCH_X86_64 OR HAHAHA_ARCH_X86_32)
    set(HAHAHA_SIMD_AVX OFF CACHE BOOL "Enable AVX/AVX2 for 256-bit SIMD (requires CPU support)")
    if (HAHAHA_SIMD_AVX)
        if (MSVC)
            target_compile_options(hahaha PUBLIC /arch:AVX2)
        else ()
            target_compile_options(hahaha PUBLIC -mavx2 -mfma)
        endif ()
    endif ()
    # SSE2 is default on x86-64; x86-32 may need -msse2
    if (HAHAHA_ARCH_X86_32 AND NOT MSVC)
        target_compile_options(hahaha PUBLIC -msse2)
    endif ()
endif ()

# dl (Linux/Unix) - harmless elsewhere.
if (CMAKE_DL_LIBS)
    target_link_libraries(hahaha PUBLIC ${CMAKE_DL_LIBS})
endif ()

# std::stacktrace may require extra libs on some gcc/libstdc++ toolchains.
foreach (_libname IN ITEMS stdc++_libbacktrace stdc++exp backtrace)
    find_library(HAHAHA_STACKTRACE_${_libname} NAMES ${_libname})
    if (HAHAHA_STACKTRACE_${_libname})
        target_link_libraries(hahaha PUBLIC ${HAHAHA_STACKTRACE_${_libname}})
    endif ()
endforeach ()

# ---- Dependencies (Full Configuration) ----
# All dependencies are managed via vcpkg.json and will be automatically installed

# CUDA: when enabled, add toolkit include path so __has_include(<driver_types.h>) works in .cpp
if (HAHAHA_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(hahaha PUBLIC CUDA::cudart)
endif ()

# OpenGL
find_package(OpenGL REQUIRED)

# GLFW3 (via vcpkg)
find_package(glfw3 CONFIG REQUIRED)
if (TARGET glfw3::glfw)
    set(_glfw_target glfw3::glfw)
elseif (TARGET glfw)
    set(_glfw_target glfw)
else ()
    message(FATAL_ERROR "GLFW3 not found. Please ensure glfw3 is installed via vcpkg.")
endif ()

# ImGui (via vcpkg)
find_package(imgui CONFIG REQUIRED)

# Link all display dependencies
target_link_libraries(hahaha PUBLIC
        ${_glfw_target}
        imgui::imgui
        OpenGL::GL
)

# ---- install ----

install(TARGETS hahaha
        EXPORT hahahaTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/core/include/"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(EXPORT hahahaTargets
        NAMESPACE h3::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hahaha"
)

# ---- subprojects ----
# Full configuration: build tests and examples
enable_testing()
add_subdirectory(tests)
#add_subdirectory(examples)
