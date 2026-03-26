# Source file collection module
# Uses regex to collect C++ and CUDA source files

function(collect_cpp_sources source_dir output_var)
    # Collect all .cpp files using GLOB_RECURSE with regex pattern
    file(GLOB_RECURSE cpp_files CONFIGURE_DEPENDS
        "${source_dir}/*.cpp"
    )
    set(${output_var} ${cpp_files} PARENT_SCOPE)
endfunction()

function(collect_cuda_sources source_dir output_var)
    # Collect all .cu files using GLOB_RECURSE with regex pattern
    file(GLOB_RECURSE cuda_files CONFIGURE_DEPENDS
        "${source_dir}/*.cu"
    )
    set(${output_var} ${cuda_files} PARENT_SCOPE)
endfunction()

function(collect_python_sources source_dir output_var)
    # Collect all Python binding source files
    file(GLOB_RECURSE python_files CONFIGURE_DEPENDS
        "${source_dir}/*.cpp"
        "${source_dir}/*.cxx"
    )
    set(${output_var} ${python_files} PARENT_SCOPE)
endfunction()

function(collect_test_sources test_dir output_var)
    # Collect all test source files with various extensions
    file(GLOB_RECURSE test_files CONFIGURE_DEPENDS
        "${test_dir}/*.cpp"
        "${test_dir}/*.cxx"
        "${test_dir}/*.cc"
    )
    set(${output_var} ${test_files} PARENT_SCOPE)
endfunction()
