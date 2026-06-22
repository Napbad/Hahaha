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

#ifndef HAHAHA_CUDAELEMENTWISE_H
#define HAHAHA_CUDAELEMENTWISE_H

#include <cuda_runtime.h>
#include <expected>
#include <string>

#include "Error.h"

namespace h3::core::compute::detail {

inline std::expected<void, Error> cudaCheck(const cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        return std::unexpected(Error(
            std::string(message) + ": " + cudaGetErrorString(err),
            ErrorCode::RuntimeError));
    }
    return {};
}

} // namespace h3::core::compute::detail

#endif // HAHAHA_CUDAELEMENTWISE_H
