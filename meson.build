project('hahaha', 'cpp',
        version : '0.0.1',
        default_options : [
            'warning_level=3',
            'cpp_std=c++23'])

include_dir = include_directories('core/include')
fs = import('fs')

# These arguments are only used to build the shared library
# not the executables that use the library.
lib_args = ['-DBUILDING_MESON_LIBRARY']

# Get all the header files
headers_raw = run_command('find', 'core/include', '-type', 'f',
                          '(' ,'-name', '*.h', '-o', '-name', '*.cuh', '-o', '-name', '*.hpp', ')',
    check: false).stdout().strip().split('\n')
headers = []
foreach h : headers_raw
    if h != ''
        headers += h
    endif
endforeach
sources = run_command('find', 'core/src', '-name', '*.cpp', '-type', 'f', check: true).stdout().strip().split('\n')

hahaha_lib = static_library('hahaha', sources,
               install : true,
               include_directories : include_dir,
               cpp_args : lib_args,
               dependencies : [
               ]
)

hahaha = declare_dependency(link_with: hahaha_lib,
                            include_directories: include_dir)


subdir('tests')
