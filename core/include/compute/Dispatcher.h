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

#ifndef HAHAHA_COMPUTE_DISPATCHER_H_IFDH89W7R89HSHFHDF
#define HAHAHA_COMPUTE_DISPATCHER_H_IFDH89W7R89HSHFHDF

#include <functional>
#include <unordered_map>
#include <string>

#include <vector>
#include "backend/Device.h"

namespace h3 {

// A very simplified dispatcher similar to c10
class Dispatcher {
public:
    using KernelFunction = std::function<void(void*, const std::vector<void*>&)>;

    static Dispatcher& get() {
        static Dispatcher instance;
        return instance;
    }

    void registerKernel(const std::string& opName, backend::DeviceType device, KernelFunction kernel) {
        kernels_[opName][device] = kernel;
    }

    KernelFunction getKernel(const std::string& opName, backend::DeviceType device) {
        if (kernels_.contains(opName)) {
            if (kernels_[opName].contains(device)) {
                return kernels_[opName][device];
            }
        }
        return nullptr;
    }

private:
    std::unordered_map<std::string, std::unordered_map<backend::DeviceType, KernelFunction>> kernels_;
};

#define H3_REGISTER_KERNEL(OpName, Device, Kernel) \
    static bool _registered_##OpName##_##Device = []() { \
        h3::Dispatcher::get().registerKernel(#OpName, Device, Kernel); \
        return true; \
    }();

} // namespace h3

#endif
