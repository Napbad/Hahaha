# tests/meson.build
test_include_dir = include_directories('../core/include')

test_sources = run_command('find', '.', '-name', '*.cpp', '-type', 'f', check: true).stdout().strip().split('\n')

gtest_dep = dependency('gtest', required: true)

test_executable = executable('hahaha_tests', test_sources,
                           include_directories: [test_include_dir, include_dir],
                           link_with: hahaha_lib,
                           dependencies: [gtest_dep])

test('hahaha_unit_tests', test_executable)

