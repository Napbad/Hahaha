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

#ifndef HAHAHA_DEFINES_H_80319B79F08F4121AFF6C3C33996DDB1
#define HAHAHA_DEFINES_H_80319B79F08F4121AFF6C3C33996DDB1
#include <cstdint>
#include <sstream>
#include <string>

namespace h3::core {
using Int8 = std::int8_t;
using Int16 = std::int16_t;
using Int32 = std::int32_t;
using Int64 = std::int64_t;

using UInt8 = std::uint8_t;
using UInt16 = std::uint16_t;
using UInt32 = std::uint32_t;
using UInt64 = std::uint64_t;

using Float32 = float;
using Float64 = double;

using SizeT = std::int64_t;


enum class Operator {
    Add, Sub, Mul, Div, Mod, Pow, Sqrt, Log, Exp, Sin, Cos, Tan, Asin, Acos, Atan,
    Abs, Sign, Ceil, Floor, Round, Trunc,
    Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Log10, Log2, Log1p, Exp2, Expm1, Cbrt,
    Erf, Erfc, Tgamma, Lgamma,
    Max, Min, Clamp,
    Count
};

enum class DataType {
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64,
    Count
};

inline SizeT sizeOf(const DataType type) {
    switch (type) {
    case DataType::Int8:
        return sizeof(Int8);
    case DataType::Int16:
        return sizeof(Int16);
    case DataType::Int32:
        return sizeof(Int32);
    case DataType::Int64:
        return sizeof(Int64);
    case DataType::UInt8:
        return sizeof(UInt8);
    case DataType::UInt16:
        return sizeof(UInt16);
    case DataType::UInt32:
        return sizeof(UInt32);
    case DataType::UInt64:
        return sizeof(UInt64);
    case DataType::Float32:
        return sizeof(Float32);
    case DataType::Float64:
        return sizeof(Float64);
    default:
        return 0;
    }
}

}

#endif // HAHAHA_DEFINES_H_80319B79F08F4121AFF6C3C33996DDB1