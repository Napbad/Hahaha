project('hahaha', 'cpp',
    version : '0.0.1',
    default_options : [
            'warning_level=3',
            'cpp_std=c++23'])

# Include directories
include_dirs = [
    'core/include',
    'extern/externlibs/imgui',
    'extern/externlibs/imgui/backends',
    'extern/externlibs/imgui/examples/libs/glfw/include' # Fallback for GLFW
]
include_dir = include_directories(include_dirs)
fs = import('fs')

# Dependencies
glfw_dep = dependency('glfw3', required : false)
gl_dep = dependency('gl', required : false)
dl_dep = meson.get_compiler('cpp').find_library('dl', required : false)

# These arguments are only used to build the shared library
lib_args = ['-DBUILDING_MESON_LIBRARY']

# Get all the header files
headers_raw = run_command('find', 'core/include', '-type', 'f',
                          '(' ,'-name', '*.h', '-o', '-name', '*.cuh', '-o', '-name', '*.hpp', ')',
                          check: false).stdout().strip().split('\n')
extern_headers_raw = run_command('find', 'extern', '-type', 'f',
                          '(' ,'-name', '*.h', '-o', '-name', '*.cuh', '-o', '-name', '*.hpp', ')',
                          '!', '-path', '*/examples/*',
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

# Conditionally add ImGui backends
imgui_backend_sources = []
# Even if dependency is not found by pkg-config, we try to compile if headers are present
# On Arch, the user might have them but pkg-config might not be set up in this environment
imgui_backend_sources += 'extern/externlibs/imgui/backends/imgui_impl_glfw.cpp'
imgui_backend_sources += 'extern/externlibs/imgui/backends/imgui_impl_opengl3.cpp'

# Exclude backends from general extern_sources to avoid duplicates
extern_sources = run_command('find', 'extern', '-name', '*.cpp', '-type', 'f',
                             '!', '-path', '*/examples/*',
                             '!', '-path', '*/backends/*',
                             '!', '-path', '*/misc/freetype/*',
                             '!', '-path', '*/misc/fonts/*',
                             check: true).stdout().strip().split('\n')

hahaha_lib = static_library('hahaha', sources + extern_sources + imgui_backend_sources,
                            install : true,
                            include_directories : include_dir,
                            cpp_args : lib_args,
                            dependencies : [
                                glfw_dep,
                                gl_dep,
                                dl_dep
                            ]
)

hahaha = declare_dependency(link_with: hahaha_lib,
                            include_directories: include_dir)

subdir('tests')
subdir('examples')
