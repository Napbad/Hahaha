project('mnist_nn', 'cpp',
        version : '0.1.0',
        default_options : [
            'warning_level=3',
            'cpp_std=c++23'])

# Include directories
mnist_include_dir = include_directories('mnist/include')
project_include_dir = include_directories('../core/include')

# Dependencies
eigen_dep = dependency('eigen3', required: false)
if not eigen_dep.found()
    # If not found through pkg-config, try using a fallback
    # You may need to adjust this path or install Eigen3
    warning('Eigen3 not found through pkg-config, trying fallback mechanism')
    eigen_dep = declare_dependency(
        include_directories: include_directories('/usr/include/eigen3')
    )
endif

# Source files for the MNIST neural network library
mnist_sources = [
    'mnist/src/dataset.cpp',
    'mnist/src/network.cpp',
    'mnist/src/layer.cpp',
    'mnist/src/activation.cpp',
    'mnist/src/loss.cpp',
    'mnist/src/optimizer.cpp',
    'mnist/src/trainer.cpp',
]

# Create the MNIST neural network library
mnist_lib = static_library('mnist_nn',
    mnist_sources,
    include_directories : [mnist_include_dir, project_include_dir],
    dependencies : [eigen_dep]
)

# Define the MNIST neural network dependency for use by executables
mnist_nn_dep = declare_dependency(
    link_with : mnist_lib,
    include_directories : [mnist_include_dir, project_include_dir],
    dependencies : [eigen_dep]
)

# Main executable
executable('mnist_recognizer',
    'mnist/src/main.cpp',
    dependencies : mnist_nn_dep,
    install : true
)
