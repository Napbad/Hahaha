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

#include "common/data/Types.h"
#include "common/macros.h"

namespace h3::backend {

using common::Bool;
using common::Float32;
using common::Float64;
using common::Int32;
using common::Int64;
using common::Int8;
using common::SizeType;
using common::UInt8;

enum class DeviceType : Int8 {
    CPU = 0,
    CUDA = 1,
};

struct Device {
    DeviceType type;
    Int32 index;

    explicit Device(const DeviceType t, const Int32 i = -1) : type(t), index(i) {
    }

    bool operator==(const Device& other) const {
        return type == other.type && index == other.index;
    }
};

enum class ScalarType : Int8 { Float, Double, Int, Long, Byte, Bool };

inline SizeType elementSize(ScalarType t) {
    switch (t) {
    case ScalarType::Float:
        return sizeof(Float32);
    case ScalarType::Double:
        return sizeof(Float64);
    case ScalarType::Int:
        return sizeof(Int32);
    case ScalarType::Long:
        return sizeof(Int64);
    case ScalarType::Byte:
        return sizeof(UInt8);
    case ScalarType::Bool:
        return sizeof(Bool);
    default:
        return 0;
    }
}

} // namespace h3::backend