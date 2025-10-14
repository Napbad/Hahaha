// Copyright (c) 2025 Napbad
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Email: napbad.sen@gmail.com
// GitHub: https://github.com/Napbad

//
// Created by Napbad on 7/27/25.
//

#ifndef H3DEFS_H
#define H3DEFS_H

namespace hahaha
{

typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned long long ullong;

typedef long long i128;
typedef long i64;
typedef int i32;
typedef short i16;
typedef char i8;

typedef float f32;
typedef double f64;

typedef unsigned long long ui128;
typedef unsigned long int ui64;
typedef unsigned int ui32;
typedef unsigned short ui16;
typedef unsigned char ui8;

typedef unsigned int sizeT;
typedef long ptrDiffT;
} // namespace hahaha

// Quick import macro for core hahaha namespaces
#define HHH_NAMESPACE_IMPORT                                                   \
    namespace hahaha                                                           \
    {                                                                          \
    }                                                                          \
    using namespace hahaha;                                                    \
    namespace hahaha::math                                                     \
    {                                                                          \
    }                                                                          \
    using namespace hahaha::math;                                              \
    namespace hahaha::core                                                     \
    {                                                                          \
    }                                                                          \
    using namespace hahaha::core;

#endif // H3DEFS_H
