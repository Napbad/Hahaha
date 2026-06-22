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

#ifndef HAHAHA_OPERATORFUNCTORS_H
#define HAHAHA_OPERATORFUNCTORS_H

#include <algorithm>
#include <cmath>
#include <type_traits>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HAHAHA_OP_HD __host__ __device__ inline
#else
#define HAHAHA_OP_HD inline
#endif

namespace h3::core::compute::detail {

template <typename T>
HAHAHA_OP_HD T toFloat(T x) {
    return x;
}

template <typename T>
HAHAHA_OP_HD auto asFloat(T x) -> std::conditional_t<std::is_floating_point_v<T>, T, Float64> {
    if constexpr (std::is_floating_point_v<T>) {
        return x;
    } else {
        return static_cast<Float64>(x);
    }
}

#define HAHAHA_UNARY_FLOAT_FUNCTOR(Name, Expr)                                           \
    struct Name##Functor {                                                               \
        template <typename T>                                                          \
        HAHAHA_OP_HD static T apply(const T x) {                                       \
            const auto v = asFloat(x);                                                   \
            if constexpr (std::is_floating_point_v<T>) {                                 \
                return static_cast<T>(Expr);                                             \
            }                                                                            \
            return static_cast<T>(static_cast<Float64>(Expr));                            \
        }                                                                                \
    }

#define HAHAHA_BINARY_FUNCTOR(Name, Expr)                                                \
    struct Name##Functor {                                                               \
        template <typename T>                                                          \
        HAHAHA_OP_HD static T apply(const T a, const T b) {                              \
            return static_cast<T>(Expr);                                                 \
        }                                                                                \
    }

HAHAHA_BINARY_FUNCTOR(Add, (a + b))
HAHAHA_BINARY_FUNCTOR(Sub, (a - b))
HAHAHA_BINARY_FUNCTOR(Mul, (a * b))

struct DivFunctor {
    template <typename T>
    HAHAHA_OP_HD static T apply(const T a, const T b) {
        return static_cast<T>(asFloat(a) / asFloat(b));
    }
};

struct ModFunctor {
    template <typename T>
    HAHAHA_OP_HD static T apply(const T a, const T b) {
        if constexpr (std::is_floating_point_v<T>) {
            return static_cast<T>(std::fmod(asFloat(a), asFloat(b)));
        }
        return static_cast<T>(a % b);
    }
};

struct PowFunctor {
    template <typename T>
    HAHAHA_OP_HD static T apply(const T a, const T b) {
        return static_cast<T>(std::pow(asFloat(a), asFloat(b)));
    }
};

struct MaxFunctor {
    template <typename T>
    HAHAHA_OP_HD static T apply(const T a, const T b) {
        return a > b ? a : b;
    }
};

struct MinFunctor {
    template <typename T>
    HAHAHA_OP_HD static T apply(const T a, const T b) {
        return a < b ? a : b;
    }
};

HAHAHA_UNARY_FLOAT_FUNCTOR(Sqrt, std::sqrt(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Log, std::log(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Exp, std::exp(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Sin, std::sin(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Cos, std::cos(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Tan, std::tan(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Asin, std::asin(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Acos, std::acos(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Atan, std::atan(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Sinh, std::sinh(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Cosh, std::cosh(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Tanh, std::tanh(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Asinh, std::asinh(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Acosh, std::acosh(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Atanh, std::atanh(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Log10, std::log10(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Log2, std::log2(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Log1p, std::log1p(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Exp2, std::exp2(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Expm1, std::expm1(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Cbrt, std::cbrt(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Erf, std::erf(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Erfc, std::erfc(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Tgamma, std::tgamma(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Lgamma, std::lgamma(v))

struct AbsFunctor {
    template <typename T>
    HAHAHA_OP_HD static T apply(const T x) {
        if constexpr (std::is_unsigned_v<T>) {
            return x;
        }
        return static_cast<T>(x < T(0) ? -x : x);
    }
};

struct SignFunctor {
    template <typename T>
    HAHAHA_OP_HD static T apply(const T x) {
        if (x > T(0)) {
            return T(1);
        }
        if (x < T(0)) {
            return T(-1);
        }
        return T(0);
    }
};

HAHAHA_UNARY_FLOAT_FUNCTOR(Ceil, std::ceil(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Floor, std::floor(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Round, std::round(v))
HAHAHA_UNARY_FLOAT_FUNCTOR(Trunc, std::trunc(v))

struct ClampFunctor {
    template <typename T>
    HAHAHA_OP_HD static T apply(const T x, const T lo, const T hi) {
        const T v = x < lo ? lo : x;
        return v > hi ? hi : v;
    }
};

#undef HAHAHA_UNARY_FLOAT_FUNCTOR
#undef HAHAHA_BINARY_FUNCTOR
#undef HAHAHA_OP_HD

} // namespace h3::core::compute::detail

#endif // HAHAHA_OPERATORFUNCTORS_H
