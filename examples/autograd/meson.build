# examples/autograd/meson.build
example_include_dir = include_directories('../../core/include')

example_sources = run_command('find', '.', '-name', '*.cpp', '-type', 'f', check: true).stdout().strip().split('\n')


example_executable = executable('hahaha_example_autograd', example_sources,
                           include_directories: [example_include_dir, include_dir],
                           link_with: hahaha_lib)

