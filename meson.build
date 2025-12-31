    project('hahaha', 'cpp',
        version : '0.0.1',
        default_options : [
                'warning_level=3',
                'cpp_std=c++23'])

include_dir = include_directories('core/include', 'extern/externlibs/imgui')
fs = import('fs')

# These arguments are only used to build the shared library
# not the executables that use the library.
lib_args = ['-DBUILDING_MESON_LIBRARY']

# Get all the header files
headers_raw = run_command('find', 'core/include', '-type', 'f',
                          '(' ,'-name', '*.h', '-o', '-name', '*.cuh', '-o', '-name', '*.hpp', ')',
                          check: false).stdout().strip().split('\n')
extern_headers_raw = run_command('find', 'extern', '-type', 'f',
                          '(' ,'-name', '*.h', '-o', '-name', '*.cuh', '-o', '-name', '*.hpp', ')',
                          '!', '-path', '*/examples/*',
                          '!', '-path', '*/backends/*',
                          '!', '-path', '*/misc/freetype/*',
                          '!', '-path', '*/misc/fonts/*',
                          check: false).stdout().strip().split('\n')
headers = []
foreach h : headers_raw
        if h != ''
                headers += h
        endif
endforeach
foreach h : extern_headers_raw
        if h != ''
                headers += h
        endif
endforeach


sources = run_command('find', 'core/src', '-name', '*.cpp', '-type', 'f', check: true).stdout().strip().split('\n')
extern_sources = run_command('find', 'extern', '-name', '*.cpp', '-type', 'f',
                             '!', '-path', '*/examples/*',
                             '!', '-path', '*/backends/*',
                             '!', '-path', '*/misc/freetype/*',
                             '!', '-path', '*/misc/fonts/*',
                             check: true).stdout().strip().split('\n')

hahaha_lib = static_library('hahaha', sources + extern_sources,
                            install : true,
                            include_directories : include_dir,
                            cpp_args : lib_args,
                            dependencies : [
                            ]
)

hahaha = declare_dependency(link_with: hahaha_lib,
                            include_directories: include_dir)


subdir('tests')
subdir('examples')