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

#include <memory>
#include <utility>
#include "common/macros.h"
#include "backend/StorageImpl.h"
#include "backend/cpu/CpuStorageImpl.h"

#ifdef HAHAHA_USE_CUDA
#include "backend/gpu/CudaStorageImpl.h"
#endif

namespace h3::backend {

class Storage {
public:
    Storage() = default;

    explicit Storage(std::shared_ptr<StorageImpl> ptr) : impl_(std::move(ptr)) {
    }

    // Convenience constructor for CPU storage
    static Storage create_cpu(size_t nbytes) {
        return Storage(std::make_shared<CpuStorageImpl>(nbytes));
    }

    [[nodiscard]] void* data() {
        return impl_ ? impl_->data() : nullptr;
    }

    [[nodiscard]] const void* data() const {
        return impl_ ? impl_->data() : nullptr;
    }

    [[nodiscard]] size_t nbytes() const {
        return impl_ ? impl_->nbytes() : 0;
    }

    [[nodiscard]] Device device() const {
        return impl_ ? impl_->device() : Device(DeviceType::CPU);
    }

    [[nodiscard]] bool defined() const {
        return impl_ != nullptr;
    }

private:
    std::shared_ptr<StorageImpl> impl_;
};

}