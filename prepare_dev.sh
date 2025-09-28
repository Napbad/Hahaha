#!/bin/bash

#
# Copyright (c) 2025 Napbad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Email: napbad.sen@gmail.com
# GitHub: https://github.com/Napbad
#
#!/bin/bash

# 容器和镜像配置
CONTAINER_NAME="hahaha_dev_container"
DOCKERFILE_PATH="Dockerfile"
IMAGE_TAG="hahaha_dev_image:latest"
WORKSPACE_DIR="$(pwd)"  # 宿主机当前目录

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 检查容器是否存在
container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# 检查容器是否正在运行
container_running() {
    docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# 获取容器使用的镜像
get_container_image() {
    docker inspect -f '{{.Config.Image}}' "${CONTAINER_NAME}" 2>/dev/null
}

# 获取Dockerfile构建的镜像ID (改进版本)
get_dockerfile_image_id() {
    # 回退到直接检查镜像标签
    local image_id=$(docker images -q "${IMAGE_TAG}" 2>/dev/null)
    if [ -n "$image_id" ]; then
        echo "$image_id"
        return 0
    fi

    # 如果镜像不存在，返回空
    return 1
}

# 构建镜像 (改进版本)
build_image() {
    echo -e "${BLUE}正在构建镜像: ${IMAGE_TAG}${NC}"

    # 尝试使用BuildKit (如果可用)
    if docker buildx version >/dev/null 2>&1; then
        echo -e "${YELLOW}使用 buildx 构建镜像${NC}"
        docker buildx build -t "${IMAGE_TAG}" -f "${DOCKERFILE_PATH}" .
    else
        echo -e "${YELLOW}使用普通 docker build 构建镜像${NC}"
        docker build -t "${IMAGE_TAG}" -f "${DOCKERFILE_PATH}" .
    fi

    if [ $? -ne 0 ]; then
        echo -e "${RED}镜像构建失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}镜像构建成功${NC}"
}

# 重新创建容器
recreate_container() {
    echo -e "${BLUE}正在重新创建容器: ${CONTAINER_NAME}${NC}"

    # 停止并删除现有容器
    if container_running; then
        echo -e "${YELLOW}停止现有容器${NC}"
        docker stop "${CONTAINER_NAME}" >/dev/null
    fi

    echo -e "${YELLOW}删除现有容器${NC}"
    docker rm "${CONTAINER_NAME}" >/dev/null

    # 创建并启动新容器
    create_and_start_container
}

# 创建并启动容器
create_and_start_container() {
    echo -e "${BLUE}正在创建并启动容器: ${CONTAINER_NAME}${NC}"

    # 启动容器，将当前目录挂载到容器的~/workspace
    docker run -d \
        --name "${CONTAINER_NAME}" \
        -v "${WORKSPACE_DIR}:/workspace" \
        -v /var/run/docker.sock:/var/run/docker.sock \
        --gpus all \
        --privileged \
        --network host \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        "${IMAGE_TAG}" \
        tail -f /dev/null

    if [ $? -ne 0 ]; then
        echo -e "${RED}容器启动失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}容器启动成功${NC}"
}

# 启动已存在的容器
start_existing_container() {
    echo -e "${BLUE}正在启动容器: ${CONTAINER_NAME}${NC}"
    docker start "${CONTAINER_NAME}"
    if [ $? -ne 0 ]; then
        echo -e "${RED}容器启动失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}容器启动成功${NC}"
}

# 进入容器
enter_container() {
    echo -e "${BLUE}正在进入容器: ${CONTAINER_NAME}${NC}"
    docker exec -it "${CONTAINER_NAME}" /bin/bash
}

# 主逻辑
main() {
    # 检查容器是否存在
    if container_exists; then
        echo -e "${YELLOW}检测到同名容器: ${CONTAINER_NAME}${NC}"

        # 检查容器是否正在运行
        if container_running; then
            echo -e "${YELLOW}容器正在运行${NC}"

            # 获取容器当前使用的镜像
            current_image=$(get_container_image)
            echo -e "${YELLOW}容器当前使用的镜像: ${current_image}${NC}"

            # 获取Dockerfile构建的镜像ID
            dockerfile_image=$(get_dockerfile_image_id)
            echo -e "${YELLOW}Dockerfile对应的镜像: ${dockerfile_image}${NC}"

            # 检查镜像是否匹配
            if [ "$current_image" = "$dockerfile_image" ]; then
                echo -e "${GREEN}镜像匹配，使用现有容器${NC}"
            else
                echo -e "${RED}镜像不匹配，更新容器${NC}"
                read -p "是否更新容器? [Y/n]: " answer
                answer=${answer:-Y}
                if [[ $answer =~ ^[Yy]$ ]]; then
                    build_image
                    recreate_container
                fi
            fi

            # 询问用户是否进入容器
            read -p "是否进入容器? [Y/n]: " answer
            answer=${answer:-Y}
            if [[ $answer =~ ^[Yy]$ ]]; then
                enter_container
            fi
        else
            echo -e "${YELLOW}容器未运行，正在启动${NC}"
            start_existing_container

            # 询问用户是否进入容器
            read -p "是否进入容器? [Y/n]: " answer
            answer=${answer:-Y}
            if [[ $answer =~ ^[Yy]$ ]]; then
                enter_container
            fi
        fi
    else
        echo -e "${YELLOW}未找到同名容器${NC}"

        # 检查是否有最新的镜像
        if docker images -q "${IMAGE_TAG}" >/dev/null 2>&1; then
            echo -e "${YELLOW}找到现有镜像: ${IMAGE_TAG}${NC}"
            create_and_start_container
        else
            echo -e "${YELLOW}未找到镜像，开始构建${NC}"
            build_image
            create_and_start_container
        fi

        # 询问用户是否进入容器
        read -p "是否进入容器? [Y/n]: " answer
        answer=${answer:-Y}
        if [[ $answer =~ ^[Yy]$ ]]; then
            enter_container
        fi
    fi
}

# 执行主逻辑
main