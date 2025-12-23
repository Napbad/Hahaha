# tests/meson.build
test_include_dir = include_directories('../core/include')

test_sources = run_command('find', '.', '-name', '*.cpp', '-type', 'f', check: true).stdout().strip().split('\n')

test_executable = executable('hahaha_tests', test_sources,
                           include_directories: [test_include_dir, include_dir],
                           link_with: hahaha_lib,
                           dependencies: [])

test('hahaha_unit_tests', test_executable)

