//  Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/jason-is-debugging/Hahaha)
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

#ifndef HAHAHA_CPUALLOCATOR_H_A89E87B09A144036BE1A4D9174F54679
#define HAHAHA_CPUALLOCATOR_H_A89E87B09A144036BE1A4D9174F54679
#include "backend/MemoryManager.h"

namespace h3::core::backend::cpu {
class CPUMemoryManager : public MemoryManager{
public:
    std::expected<CommonPointer, Error> allocate(SizeT size) override;
    void deallocate(CommonPointer ptr) override;

    ~CPUMemoryManager() override;

    std::expected<void, Error>
    move(CommonPointer dst, CommonPointer src, SizeT size) override;

    std::expected<void, Error>
    copy(CommonPointer dst, CommonPointer src, SizeT size) override;

    std::expected<void, Error> copyFromHostToDevice(CommonPointer dst,
        CommonPointer src,
        SizeT size) override;

    std::expected<void, Error> copyFromDeviceToHost(CommonPointer dst,
        CommonPointer src,
        SizeT size) override;
};
}

#endif // HAHAHA_CPUALLOCATOR_H_A89E87B09A144036BE1A4D9174F54679
