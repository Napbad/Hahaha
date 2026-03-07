// Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//

#pragma once

#include "backend/StorageImpl.h"
#include <cstdlib>
#include <new>

namespace h3::backend {

// CPU Storage Implementation
class CpuStorageImpl : public StorageImpl {
public:
    explicit CpuStorageImpl(const size_t size) : size_(size), owns_memory_(true) {
        ptr_ = std::malloc(size);
        if (!ptr_ && size > 0) {
            throw std::bad_alloc();
        }
    }

    // For wrapping existing memory
    CpuStorageImpl(void* ptr, const size_t size, const bool take_ownership = false)
        : ptr_(ptr), size_(size), owns_memory_(take_ownership) {}

    ~CpuStorageImpl() override {
        if (owns_memory_ && ptr_) {
            std::free(ptr_);
        }
    }

    // Delete copy constructor and assignment to prevent double free
    CpuStorageImpl(const CpuStorageImpl&) = delete;
    CpuStorageImpl& operator=(const CpuStorageImpl&) = delete;

    // Allow move
    CpuStorageImpl(CpuStorageImpl&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_), owns_memory_(other.owns_memory_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.owns_memory_ = false;
    }

    CpuStorageImpl& operator=(CpuStorageImpl&& other) noexcept {
        if (this != &other) {
            if (owns_memory_ && ptr_) {
                std::free(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            owns_memory_ = other.owns_memory_;
            other.ptr_ = nullptr;
            other.size_ = 0;
            other.owns_memory_ = false;
        }
        return *this;
    }

    [[nodiscard]] void* data() override { return ptr_; }
    [[nodiscard]] const void* data() const override { return ptr_; }
    [[nodiscard]] size_t nbytes() const override { return size_; }
    [[nodiscard]] Device device() const override { return Device(DeviceType::CPU); }

private:
    void* ptr_;
    size_t size_;
    bool owns_memory_;
};

}