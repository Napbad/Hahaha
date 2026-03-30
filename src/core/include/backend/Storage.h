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

#ifndef HAHAHA_STORAGE_H_70F22415576A41E6913976C1B6331B99
#define HAHAHA_STORAGE_H_70F22415576A41E6913976C1B6331B99
#include <memory>

#include "MemoryManager.h"
#include "Device.h"

namespace h3::core::backend {

class Storage {
public:
    Storage(const CommonPointer& data,
            const std::shared_ptr<MemoryManager>& manager,
            const SizeT size,
            const bool isView) : m_data(data), m_memoryManager(manager),
                                 m_size(size), m_isView(isView) {
    };

    Storage() : m_data(CommonPointer()), m_memoryManager(getDefaultMemoryManager()),
                m_size(0), m_isView(false) {

    }

    ~Storage() {
        if (!m_isView) {
            m_memoryManager->deallocate(m_data);
        }
    }

    [[nodiscard]] CommonPointer data() const {
        return m_data;
    }

    [[nodiscard]] std::shared_ptr<MemoryManager> memoryManager() const {
        return m_memoryManager;
    }

    [[nodiscard]] SizeT size() const {
        return m_size;
    }

    [[nodiscard]] Device device() const {
        return m_data.device();
    }

    [[nodiscard]] Storage createView() const;

    [[nodiscard]] std::expected<Storage, Error> clone() const;

    std::expected<CommonPointer, Error> resize(SizeT size);

private:
    CommonPointer m_data;
    std::shared_ptr<MemoryManager> m_memoryManager;
    SizeT m_size;
    bool m_isView;
};
}

#endif // HAHAHA_STORAGE_H_70F22415576A41E6913976C1B6331B99