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
// Created by napbad on 3/27/26.
//

#include <memory>
#include <unordered_map>

#include "backend/MemoryManager.h"

#include "backend/cpu/CPUMemoryManager.h"

namespace h3::core::backend {

// Out-of-line virtuals so the base vtable and RTTI are emitted in this translation unit.
std::expected<void, Error> MemoryManager::move(
    CommonPointer dst,
    CommonPointer src,
    const SizeT size) {
    (void)dst;
    (void)src;
    (void)size;
    return std::unexpected(Error(
        "MemoryManager::move not implemented for this backend",
        ErrorCode::BaseError));
}

std::expected<void, Error> MemoryManager::copy(
    CommonPointer dst,
    CommonPointer src,
    const SizeT size) {
    (void)dst;
    (void)src;
    (void)size;
    return std::unexpected(Error(
        "MemoryManager::copy not implemented for this backend",
        ErrorCode::BaseError));
}

std::expected<void, Error> MemoryManager::copyFromHostToDevice(
    CommonPointer dst,
    CommonPointer src,
    const SizeT size) {
    (void)dst;
    (void)src;
    (void)size;
    return std::unexpected(Error(
        "MemoryManager::copyFromHostToDevice not implemented for this backend",
        ErrorCode::BaseError));
}

std::expected<void, Error> MemoryManager::copyFromDeviceToHost(
    CommonPointer dst,
    CommonPointer src,
    const SizeT size) {
    (void)dst;
    (void)src;
    (void)size;
    return std::unexpected(Error(
        "MemoryManager::copyFromDeviceToHost not implemented for this backend",
        ErrorCode::BaseError));
}

static std::shared_ptr<MemoryManager> defaultMemoryManager = std::make_shared<
    cpu::CPUMemoryManager>();

static std::unordered_map<Device, std::shared_ptr<MemoryManager>, DeviceHash>
memoryManagerMap;

std::shared_ptr<MemoryManager> getDefaultMemoryManager() {
    return defaultMemoryManager;
}

std::shared_ptr<MemoryManager> getMemoryManagerOn(const Device device) {
    if (const auto it = memoryManagerMap.find(device); it != memoryManagerMap.
        end()) {
        return it->second;
    }
    return nullptr;
}

void registerMemoryManager(const Device device,
                           const std::shared_ptr<MemoryManager>& manager) {
    memoryManagerMap[device] = manager;
}
}