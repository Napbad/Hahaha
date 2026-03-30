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

#include "backend/cpu/CPUMemoryManager.h"

#include <cstring>


namespace h3::core::backend::cpu {
std::expected<CommonPointer, Error> CPUMemoryManager::allocate(const SizeT size) {
    return CommonPointer{new char[size], Device{0, DeviceType::CPU}, size};
}

void CPUMemoryManager::deallocate(const CommonPointer ptr) {
    return delete []ptr.as<char>();
}

CPUMemoryManager::~CPUMemoryManager() = default;

std::expected<void, Error> CPUMemoryManager::move(CommonPointer dst,
    CommonPointer src,
    SizeT size) {
    if (dst.size() < size || src.size() < size) {
        return std::unexpected(Error("Buffer size insufficient for move operation", 
            ErrorCode::MemoryError));
    }
    
    std::memmove(dst.as<void>(), src.as<const void>(), size);
    return {};
}

std::expected<void, Error> CPUMemoryManager::copy(CommonPointer dst,
    CommonPointer src,
    SizeT size) {
    if (dst.size() < size || src.size() < size) {
        return std::unexpected(Error("Buffer size insufficient for copy operation", 
            ErrorCode::MemoryError));
    }
    
    // For CPU, copy is memcpy (assumes non-overlapping regions)
    std::memcpy(dst.as<void>(), src.as<const void>(), size);
    return {};
}

std::expected<void, Error> CPUMemoryManager::copyFromHostToDevice(CommonPointer dst,
    CommonPointer src,
    SizeT size) {
    // For CPU backend, host and device are the same, so just use copy
    return copy(dst, src, size);
}

std::expected<void, Error> CPUMemoryManager::copyFromDeviceToHost(CommonPointer dst,
    CommonPointer src,
    SizeT size) {
    // For CPU backend, host and device are the same, so just use copy
    return copy(dst, src, size);
}
}