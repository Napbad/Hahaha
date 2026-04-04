//  Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//       https://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Contributors:
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//

//
// Created by napbad on 3/26/26.
//

#ifdef HAHAHA_ENABLE_CUDA

#include <cuda_runtime.h>

#include "backend/cuda/CUDAMemoryManager.h"

namespace h3::core::backend::cuda {

CUDAMemoryManager::~CUDAMemoryManager() = default;

std::expected<CommonPointer, Error> CUDAMemoryManager::allocate(const SizeT size) {
    void* device_ptr = nullptr;
    cudaError_t err = cudaMalloc(&device_ptr, size);
    
    if (err != cudaSuccess) {
        return std::unexpected(Error(
            std::string("CUDA memory allocation failed: ") + cudaGetErrorString(err),
            ErrorCode::MemoryError));
    }
    
    return CommonPointer{device_ptr, Device{0, DeviceType::CUDA}, size};
}

void CUDAMemoryManager::deallocate(CommonPointer ptr) {
    if (ptr.get() != nullptr) {
        cudaFree(ptr.get());
    }
}

std::expected<void, Error> CUDAMemoryManager::move(CommonPointer dst,
    CommonPointer src,
    SizeT size) {
    if (dst.size() < size || src.size() < size) {
        return std::unexpected(Error("Buffer size insufficient for move operation", 
            ErrorCode::MemoryError));
    }
    
    // For CUDA device memory, use cudaMemcpy with appropriate kind based on device type
    // Since both are on CUDA, use device-to-device copy
    cudaError_t err = cudaMemcpy(dst.as<void>(), src.as<const void>(), size, cudaMemcpyDeviceToDevice);
    
    if (err != cudaSuccess) {
        return std::unexpected(Error(
            std::string("CUDA move operation failed: ") + cudaGetErrorString(err),
            ErrorCode::MemoryError));
    }
    
    return {};
}

std::expected<void, Error> CUDAMemoryManager::copy(CommonPointer dst,
    CommonPointer src,
    SizeT size) {
    if (dst.size() < size || src.size() < size) {
        return std::unexpected(Error("Buffer size insufficient for copy operation", 
            ErrorCode::MemoryError));
    }
    
    // Same as move for CUDA - device to device copy
    return move(dst, src, size);
}

std::expected<void, Error>
CUDAMemoryManager::copyFromHostToDevice(CommonPointer dst,
    CommonPointer src,
    SizeT size) {
    if (dst.size() < size || src.size() < size) {
        return std::unexpected(Error("Buffer size insufficient for host-to-device copy", 
            ErrorCode::MemoryError));
    }
    
    cudaError_t err = cudaMemcpy(dst.as<void>(), src.as<const void>(), size, cudaMemcpyHostToDevice);
    
    if (err != cudaSuccess) {
        return std::unexpected(Error(
            std::string("CUDA host-to-device copy failed: ") + cudaGetErrorString(err),
            ErrorCode::MemoryError));
    }
    
    return {};
}

std::expected<void, Error>
CUDAMemoryManager::copyFromDeviceToHost(CommonPointer dst,
    CommonPointer src,
    SizeT size) {
    if (dst.size() < size || src.size() < size) {
        return std::unexpected(Error("Buffer size insufficient for device-to-host copy", 
            ErrorCode::MemoryError));
    }
    
    cudaError_t err = cudaMemcpy(dst.as<void>(), src.as<const void>(), size, cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess) {
        return std::unexpected(Error(
            std::string("CUDA device-to-host copy failed: ") + cudaGetErrorString(err),
            ErrorCode::MemoryError));
    }
    
    return {};
}

} // namespace h3::core::backend::cuda

#endif