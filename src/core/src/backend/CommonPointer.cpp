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

#include "backend/CommonPointer.h"

#include <memory>

#include "backend/MemoryManager.h"
#include "utils/handler/exception_handler.h"

namespace h3::core::backend {

void CommonPointer::destroy() {
    if (m_ptr == nullptr) {
        return;
    }
    if (std::shared_ptr<MemoryManager> mgr = getMemoryManagerOn(m_device)) {
        mgr->deallocate(*this);
        return;
    }
    if (m_device.type() == DeviceType::CPU && m_device.index() == 0) {
        getDefaultMemoryManager()->deallocate(*this);
        return;
    }
    ThrowRuntime("no suitable deallocator found for device {}", m_device.toString());
}

}