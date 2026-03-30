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

#ifndef HAHAHA_ALLOCATOR_H_66B60E809AE7413F9740BCAF7977C1EC
#define HAHAHA_ALLOCATOR_H_66B60E809AE7413F9740BCAF7977C1EC
#include <expected>
#include <memory>

#include "CommonPointer.h"
#include "defines.h"
#include "Error.h"

namespace h3::core::backend {
// virtual MemoryManager
class MemoryManager {
public:
    virtual ~MemoryManager() = default;

    virtual std::expected<CommonPointer, Error> allocate(SizeT size) = 0;

    virtual void deallocate(CommonPointer ptr) = 0;

    virtual std::expected<void, Error> move(CommonPointer dst, CommonPointer src, SizeT size);

    virtual std::expected<void, Error> copy(CommonPointer dst, CommonPointer src, SizeT size);

    virtual std::expected<void, Error> copyFromHostToDevice(
        CommonPointer dst,
        CommonPointer src,
        SizeT size
        );

    virtual std::expected<void, Error> copyFromDeviceToHost(
        CommonPointer dst,
        CommonPointer src,
        SizeT size
        );
};

std::shared_ptr<MemoryManager> getDefaultMemoryManager();

std::shared_ptr<MemoryManager> getMemoryManagerOn(Device device);

void registerMemoryManager(Device device, const std::shared_ptr<MemoryManager>& manager);

}

#endif // HAHAHA_ALLOCATOR_H_66B60E809AE7413F9740BCAF7977C1EC