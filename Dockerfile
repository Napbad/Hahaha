ARG IMAGE_NAME=nvidia/cuda
ARG TARGETARCH=amd64

FROM ${IMAGE_NAME}:13.0.0-runtime-ubuntu24.04 AS base

ENV NV_CUDA_LIB_VERSION="13.0.0-1"

FROM base AS base-amd64

ENV NV_CUDA_CUDART_DEV_VERSION=13.0.48-1
ENV NV_NVML_DEV_VERSION=13.0.39-1
ENV NV_LIBCUSPARSE_DEV_VERSION=12.6.2.49-1
ENV NV_LIBNPP_DEV_VERSION=13.0.0.50-1
ENV NV_LIBNPP_DEV_PACKAGE=libnpp-dev-13-0=${NV_LIBNPP_DEV_VERSION}

ENV NV_LIBCUBLAS_DEV_VERSION=13.0.0.19-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME=libcublas-dev-13-0
ENV NV_LIBCUBLAS_DEV_PACKAGE=${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}

ENV NV_CUDA_NSIGHT_COMPUTE_VERSION=13.0.0-1
ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE=cuda-nsight-compute-13-0=${NV_CUDA_NSIGHT_COMPUTE_VERSION}

ENV NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION=2.27.7-1
ENV NCCL_VERSION=2.27.7-1
ENV NV_LIBNCCL_DEV_PACKAGE=${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda13.0

FROM base AS base-arm64

ENV NV_CUDA_CUDART_DEV_VERSION=13.0.48-1
ENV NV_NVML_DEV_VERSION=13.0.39-1
ENV NV_LIBCUSPARSE_DEV_VERSION=12.6.2.49-1
ENV NV_LIBNPP_DEV_VERSION=13.0.0.50-1
ENV NV_LIBNPP_DEV_PACKAGE=libnpp-dev-13-0=${NV_LIBNPP_DEV_VERSION}

ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME=libcublas-dev-13-0
ENV NV_LIBCUBLAS_DEV_VERSION=13.0.0.19-1
ENV NV_LIBCUBLAS_DEV_PACKAGE=${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}

ENV NV_CUDA_NSIGHT_COMPUTE_VERSION=13.0.0-1
ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE=cuda-nsight-compute-13-0=${NV_CUDA_NSIGHT_COMPUTE_VERSION}

ENV NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION=2.27.7-1
ENV NCCL_VERSION=2.27.7-1
ENV NV_LIBNCCL_DEV_PACKAGE=${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda13.0

FROM base-${TARGETARCH}

LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-dev-13-0=${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-command-line-tools-13-0=${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-13-0=${NV_CUDA_LIB_VERSION} \
    cuda-libraries-dev-13-0=${NV_CUDA_LIB_VERSION} \
    cuda-nvml-dev-13-0=${NV_NVML_DEV_VERSION} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    libcusparse-dev-13-0=${NV_LIBCUSPARSE_DEV_VERSION} \
    ${NV_LIBCUBLAS_DEV_PACKAGE} \
    ${NV_LIBNCCL_DEV_PACKAGE} \
    ${NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME}
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gdb \
    git \
    wget \
    curl \
    libcurl4-openssl-dev \
    libc-bin \
    vim \
    sudo \
    python3 \
    swig \
    python3-pip \
    libgtest-dev \
    clang \
    meson \
    fish \
    clangd \
    clang-format\
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    pkgconf \
    && rm -rf /var/lib/apt/lists/*

# Make clang/clang++ the system default C/C++ compilers so tools like Meson
# which consult `cc`/`c++` or $CC/$CXX will use clang inside the container.
# We register both clang and gcc with update-alternatives and set clang as the
# higher-priority choice.
RUN update-alternatives --install /usr/bin/cc cc /usr/bin/clang 100 \
 && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 50 \
 && update-alternatives --set cc /usr/bin/clang \
 && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100 \
 && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 50 \
 && update-alternatives --set c++ /usr/bin/clang++ || true

# Also set environment variables so build systems that respect CC/CXX prefer
# clang/clang++ by default. Project-level settings or Meson env overrides
# can still override these.
ENV CC=clang
ENV CXX=clang++

# Build and install Google Test from source
# The Debian/Ubuntu package `libgtest-dev` provides sources under /usr/src.
# Use the standard location installed by the package instead of a non-existent /usr/core path.
RUN if [ -d "/usr/src/googletest" ]; then \
            cd /usr/src/googletest && \
            cmake . && \
            cmake --build . --target install; \
        elif [ -d "/usr/src/gtest" ]; then \
            cd /usr/src/gtest && \
            cmake . && \
            cmake --build . --target install; \
        else \
            echo "googletest sources not found in /usr/src; skipping build"; \
        fi

# RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /workspace

# RUN cd /workspace/core/python_interface/
# RUN uv sync

CMD ["/bin/fish"]

