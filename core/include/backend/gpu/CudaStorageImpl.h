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

#include "common/macros.h"

#ifdef HAHAHA_USE_CUDA

#include "backend/StorageImpl.h"
#include <stdexcept>

// Forward declaration to avoid including cuda headers everywhere
extern "C" {
// These are standard CUDA runtime functions, but we declare them here
// to avoid heavy includes if not necessary.
// In a real project, you might wrap these in a CudaUtils header.
using cudaError_t = int;
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaSetDevice(int device);
}

namespace h3::backend {

// CUDA Storage Implementation
class CudaStorageImpl : public StorageImpl {
public:
    explicit CudaStorageImpl(const size_t size, const int device_index = 0)
        : size_(size), device_index_(device_index), owns_memory_(true) {

        if (size > 0) {
            cudaSetDevice(device_index);
            if (cudaMalloc(&ptr_, size) != 0) {
                // 0 is cudaSuccess
                throw std::runtime_error("CUDA out of memory");
            }
        } else {
            ptr_ = nullptr;
        }
    }

    // For wrapping existing GPU memory
    CudaStorageImpl(void* ptr,
                    const size_t size,
                    const int device_index,
                    const bool take_ownership = false)
        : ptr_(ptr), size_(size), device_index_(device_index),
          owns_memory_(take_ownership) {
    }

    ~CudaStorageImpl() override {
        if (owns_memory_ && ptr_) {
            cudaFree(ptr_);
        }
    }

    // Delete copy constructor and assignment
    CudaStorageImpl(const CudaStorageImpl&) = delete;
    CudaStorageImpl& operator=(const CudaStorageImpl&) = delete;

    // Allow move
    CudaStorageImpl(CudaStorageImpl&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_),
          device_index_(other.device_index_), owns_memory_(other.owns_memory_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.owns_memory_ = false;
    }

    CudaStorageImpl& operator=(CudaStorageImpl&& other) noexcept {
        if (this != &other) {
            if (owns_memory_ && ptr_) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            device_index_ = other.device_index_;
            owns_memory_ = other.owns_memory_;
            other.ptr_ = nullptr;
            other.size_ = 0;
            other.owns_memory_ = false;
        }
        return *this;
    }

    [[nodiscard]] void* data() override {
        return ptr_;
    }

    [[nodiscard]] const void* data() const override {
        return ptr_;
    }

    [[nodiscard]] size_t nbytes() const override {
        return size_;
    }

    [[nodiscard]] Device device() const override {
        return Device(DeviceType::CUDA, device_index_);
    }

private:
    void* ptr_;
    size_t size_;
    int device_index_;
    bool owns_memory_;
};

}

#endif