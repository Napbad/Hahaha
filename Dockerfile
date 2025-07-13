FROM docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04
LABEL authors="napbad"

# 安装构建依赖（保持root权限）
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc-10 g++-10 \
    git \
    vim \
    wget \
    gdb \
    googletest \
    googletest-tools

# 创建Ubuntu用户和环境配置（可选，保持root用户作为默认）
RUN useradd -m -s /bin/bash ubuntu && \
    echo "ubuntu:ubuntu" | chpasswd && \
    adduser ubuntu sudo && \
    mkdir -p /home/ubuntu && \
    chown -R ubuntu:ubuntu /home/ubuntu

# 设置工作目录
WORKDIR /workspace
