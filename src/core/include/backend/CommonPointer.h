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
// Created by napbad on 3/27/26.
//

#ifndef HAHAHA_COMMONPOINTER_H_58ECCE0C996E46FFAE684418FECD1516
#define HAHAHA_COMMONPOINTER_H_58ECCE0C996E46FFAE684418FECD1516

#include <string>

#include "Device.h"

namespace h3::core::backend {
class CommonPointer {
public:
    // Construct from raw pointer + device info
    CommonPointer(void* ptr, const Device device, const SizeT sizeInBytes = 0)
        : m_ptr(ptr), m_device(device), m_size(sizeInBytes) {
    }

    CommonPointer() : m_ptr(nullptr), m_device(Device()), m_size(0) {

    }

    [[nodiscard]] void* get() const {
        return m_ptr;
    }

    [[nodiscard]] Device device() const {
        return m_device;
    }

    [[nodiscard]] SizeT size() const {
        return m_size;
    }

    // Helper to cast safely
    template <typename T>
    T* as() const {
        return static_cast<T*>(m_ptr);
    }

    // Nice for debugging
    [[nodiscard]] std::string to_string() const {
        return "CommonPointer{" + std::to_string(
                reinterpret_cast<std::uint64_t>(m_ptr)) + ", " +
            device().toString()
            + ", " + std::to_string(m_size) + "}";
    }

    CommonPointer operator+(const std::ptrdiff_t offsetInChar) const {
        return {
            static_cast<void*>(static_cast<char*>(m_ptr) + offsetInChar),
            m_device,
            m_size
        };
    }

    bool operator==(const CommonPointer& other) const {
        return m_ptr == other.m_ptr && m_device == other.m_device;
    }

    bool operator!=(const CommonPointer& other) const {
        return m_ptr != other.m_ptr || m_device != other.m_device;
    }

    bool operator==(std::nullptr_t) const {
        return m_ptr == nullptr;
    }

    /// Releases storage via the \ref MemoryManager for \ref device() 
    void destroy();

private:
    void* m_ptr;
    Device m_device;
    SizeT m_size;
};

}


#endif //HAHAHA_COMMONPOINTER_H_58ECCE0C996E46FFAE684418FECD1516