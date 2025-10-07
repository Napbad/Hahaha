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
// Created by napbad on 10/6/25.
//

#ifndef HAHAHA_MATHFN_H
#define HAHAHA_MATHFN_H
#include "common/defines/h3defs.h"
#include <cmath>

HHHNamespaceImport

    namespace hahaha::math {
    class Math {
    public:
        static constexpr f64 pi = 3.14159265358979323846;
        static constexpr f64 e = 2.71828182845904523536;
        static constexpr f64 sqrt2 = 1.41421356237309504880;
        static constexpr f64 sqrt3 = 1.73205080756887729352;
        static constexpr f64 sqrt5 = 2.23606797749978969640;
        static constexpr f64 sqrt7 = 2.64575131106459059050;

        static f32 exp (const f32 x) {
            return expf(x);
        }
        static f64 exp (const f64 x) {
            return ::exp(x);
        }

        static f32 log (const f32 x) {
            return logf(x);
        }
        static f64 log (const f64 x) {
            return ::log(x);
        }
        static f32 log10 (const f32 x) {
            return log10f(x);
        }
        static f64 log10 (const f64 x) {
            return ::log10(x);
        }
        static f32 log2 (const f32 x) {
            return log2f(x);
        }
        static f64 log2 (const f64 x) {
            return ::log2(x);
        }
        static f32 log1p (const f32 x) {
            return log1pf(x);
        }
        static f64 log1p (const f64 x) {
            return ::log1p(x);
        }
        static f32 logb (const f32 x) {
            return logbf(x);
        }
        static f64 logb (const f64 x) {
            return ::logb(x);
        }
        static f32 tanh(const f32 x) {
            return tanhf(x);
        }
        static f64 tanh(const f64 x) {
            return ::tanh(x);
        }
    };
}

#endif // HAHAHA_MATHFN_H
