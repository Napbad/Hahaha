file(GLOB_RECURSE HAHAHA_TEST_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# ---- GoogleTest dependency ----
set(_gtest_main_target "")

find_package(GTest CONFIG QUIET) # vcpkg/homebrew usually provide CONFIG packages
if(NOT GTest_FOUND)
  find_package(GTest QUIET)      # fallback to CMake's FindGTest module (system installs)
endif()
if(TARGET GTest::gtest_main)
  set(_gtest_main_target GTest::gtest_main)
elseif(TARGET GTest::gtest)
  # Some distros don't ship gtest_main as an imported target; fall back.
  set(_gtest_main_target GTest::gtest)
endif()

if(NOT _gtest_main_target)
  message(FATAL_ERROR "GTest dependency not found.  Install GTest via repo-local vcpkg (preferred) or system GTest.")
endif()

if (HAHAHA_USE_CUDA)
  message("Using cuda to compile")
  # CUDA host_config.h only officially supports VS 2019/2022; newer MSVC needs this flag
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-Bdynamic")

  find_package(CUDAToolkit REQUIRED)
endif ()

add_executable(hahaha_tests ${HAHAHA_TEST_SOURCES})
target_link_libraries(hahaha_tests PRIVATE hahaha::hahaha ${_gtest_main_target})

# Downgrade C++ standard for tests to match pre-built gtest library (avoids linker errors like unresolved std::rotate)
set_target_properties(hahaha_tests PROPERTIES CXX_STANDARD 20)

# Optional coverage flags (gcc/clang).
if(HAHAHA_ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hahaha_tests PRIVATE -O0 -g --coverage)
    target_link_options(hahaha_tests PRIVATE --coverage)
  endif()
endif()

if(HAHAHA_USE_CUDA)
  target_compile_definitions(hahaha_tests PUBLIC HAHAHA_USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  # Enable separable compilation if you use device linking or dynamic parallelism
  set_target_properties(hahaha_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  # Link against common CUDA libraries if needed (e.g. cuBLAS, cuFFT)
  target_link_libraries(hahaha_tests PRIVATE CUDA::cublas CUDA::cudart)
endif()

# MSVC: large translation units from typed tests may require /bigobj.
if(MSVC)
  target_compile_options(hahaha_tests PRIVATE /bigobj)
endif()

add_test(NAME hahaha_unit_tests COMMAND hahaha_tests)
