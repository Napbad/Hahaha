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

CONTAINER_NAME="hahaha_dev_container"
DOCKERFILE_PATH="Dockerfile"
IMAGE_TAG="hahaha_dev"
WORKSPACE_DIR="$(pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' 

container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

container_running() {
    docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

get_container_image() {
    # Return the image ID (sha) used by the container if it exists
    docker inspect -f '{{.Image}}' "${CONTAINER_NAME}" 2>/dev/null
}

get_dockerfile_image_id() {
    local image_id=$(docker images -q "${IMAGE_TAG}" 2>/dev/null)
    if [ -n "$image_id" ]; then
        echo "$image_id"
        return 0
    fi

    return 1
}

build_image() {
    echo -e "${BLUE}Building image: ${IMAGE_TAG}${NC}"

    if docker buildx version >/dev/null 2>&1; then
        echo -e "${YELLOW}Using buildx to build image${NC}"
        # Use --load so the built image is available to the local docker daemon
        # and pass host user args so the image user matches the host
        docker buildx build --load -t "${IMAGE_TAG}" -f "${DOCKERFILE_PATH}" \
            --build-arg USERNAME="$(id -un)" --build-arg USER_UID="$(id -u)" --build-arg USER_GID="$(id -g)" .
    else
        echo -e "${YELLOW}Using docker build to build image${NC}"
        docker build -t "${IMAGE_TAG}" -f "${DOCKERFILE_PATH}" \
            --build-arg USERNAME="$(id -un)" --build-arg USER_UID="$(id -u)" --build-arg USER_GID="$(id -g)" .
    fi

    if [ $? -ne 0 ]; then
        echo -e "${RED}Image build failed${NC}"
        exit 1
    fi
    echo -e "${GREEN}Image build succeeded${NC}"
}

recreate_container() {
    echo -e "${BLUE}Recreating container: ${CONTAINER_NAME}${NC}"

    if container_running; then
        echo -e "${YELLOW}Stopping existing container${NC}"
        docker stop "${CONTAINER_NAME}" >/dev/null
    fi

    echo -e "${YELLOW}Removing existing container${NC}"
    docker rm "${CONTAINER_NAME}" >/dev/null

    # Create and start a new container
    create_and_start_container
}

# 创建并启动容器
create_and_start_container() {
    echo -e "${BLUE}Creating and starting container: ${CONTAINER_NAME}${NC}"

    # Start the container, mount the current directory to /workspace
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
        echo -e "${RED}Failed to start container${NC}"
        exit 1
    fi
    echo -e "${GREEN}Container started successfully${NC}"
}

# 启动已存在的容器
start_existing_container() {
    echo -e "${BLUE}Starting container: ${CONTAINER_NAME}${NC}"
    docker start "${CONTAINER_NAME}"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to start container${NC}"
        exit 1
    fi
    echo -e "${GREEN}Container started successfully${NC}"
}

# 进入容器
enter_container() {
    echo -e "${BLUE}Entering container: ${CONTAINER_NAME}${NC}"
    docker exec -it "${CONTAINER_NAME}" /bin/bash
}

# 主逻辑
main() {
    # Check whether the container exists
    if container_exists; then
        echo -e "${YELLOW}Found existing container: ${CONTAINER_NAME}${NC}"

        # Check whether it's running
        if container_running; then
            echo -e "${YELLOW}Container is currently running${NC}"

            # Get image ID used by the running container
            current_image_id=$(get_container_image)
            echo -e "${YELLOW}Container image ID: ${current_image_id}${NC}"

            # Get image ID built from Dockerfile (if present)
            dockerfile_image_id=$(get_dockerfile_image_id)
            echo -e "${YELLOW}Dockerfile image ID: ${dockerfile_image_id}${NC}"

            # Compare image IDs
            if [ -n "$dockerfile_image_id" ] && [ "$current_image_id" = "$dockerfile_image_id" ]; then
                echo -e "${GREEN}Image matches container; using existing container${NC}"
            else
                echo -e "${RED}Image mismatch or Dockerfile image not found; updating container${NC}"
                read -p "Update container? [Y/n]: " answer
                answer=${answer:-Y}
                if [[ $answer =~ ^[Yy]$ ]]; then
                    build_image
                    recreate_container
                fi
            fi

            # Ask whether to enter the container
            read -p "Enter container now? [Y/n]: " answer
            answer=${answer:-Y}
            if [[ $answer =~ ^[Yy]$ ]]; then
                enter_container
            fi
        else
            echo -e "${YELLOW}Container is not running; starting it${NC}"
            start_existing_container

            # Ask whether to enter the container
            read -p "Enter container now? [Y/n]: " answer
            answer=${answer:-Y}
            if [[ $answer =~ ^[Yy]$ ]]; then
                enter_container
            fi
        fi
    else
        echo -e "${YELLOW}No existing container found${NC}"

        # If image exists locally, create the container; otherwise build then create
        # Note: `docker run` will attempt to pull the image from a registry if it
        # is not present locally. We check explicitly for a local image here.
        if [ -n "$(docker images -q "${IMAGE_TAG}")" ]; then
            echo -e "${YELLOW}Found existing image: ${IMAGE_TAG}${NC}"
            create_and_start_container
        else
            echo -e "${YELLOW}Image not found; building${NC}"
            build_image
            create_and_start_container
        fi

        # Ask whether to enter the container
        read -p "Enter container now? [Y/n]: " answer
        answer=${answer:-Y}
        if [[ $answer =~ ^[Yy]$ ]]; then
            enter_container
        fi
    fi
}

# 执行主逻辑
main