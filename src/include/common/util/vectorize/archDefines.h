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
// Created by napbad on 9/28/25.
//

#ifndef HAHAHA_VECTORIZE_H
#define HAHAHA_VECTORIZE_H
namespace hahaha::common::util {
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

    struct Vec4f {
#if defined(__AVX__)
        __m128 data;
#elif defined(__ARM_NEON)
        f3232x4_t data;
#else
        f32 data[4];
#endif
    };

    inline Vec4f vadd(const Vec4f& a, const Vec4f& b) {
        Vec4f r{};
#if defined(__AVX__)
        r.data = _mm_add_ps(a.data, b.data);
#elif defined(__ARM_NEON)
        r.data = vaddq_f32(a.data, b.data);
#else
        for (int i = 0; i < 4; i++) {
            r.data[i] = a.data[i] + b.data[i];
        }
#endif
        return r;
    }

    inline Vec4f vmul(const Vec4f& a, const Vec4f& b) {
        Vec4f r{};
#if defined(__AVX__)
        r.data = _mm_mul_ps(a.data, b.data);
#elif defined(__ARM_NEON)
        r.data = vmulq_f32(a.data, b.data);
#else
        for (int i = 0; i < 4; i++) {
            r.data[i] = a.data[i] * b.data[i];
        }
#endif
        return r;
    }
    inline Vec4f vsub(const Vec4f& a, const Vec4f& b) {
        Vec4f r{};
#if defined(__AVX__)
        r.data = _mm_sub_ps(a.data, b.data);
#elif defined(__ARM_NEON)
        r.data = vsubq_f32(a.data, b.data);
#else
        for (int i = 0; i < 4; i++) {
            r.data[i] = a.data[i] - b.data[i];
        }
#endif
        return r;
    }

    inline Vec4f vdiv(const Vec4f& a, const Vec4f& b) {
        Vec4f r{};
#if defined(__AVX__)
        r.data = _mm_div_ps(a.data, b.data);
#elif defined(__ARM_NEON)
        r.data = vdivq_f32(a.data, b.data);
#else
        for (int i = 0; i < 4; i++) {
            r.data[i] = a.data[i] / b.data[i];
        }
#endif
        return r;
    }

    inline void add_arrays(const f32* a, const f32* b, f32* out, size_t n) {
        size_t i = 0;

        // process 4 elements at a time
        for (; i + 4 <= n; i += 4) {
            Vec4f va{}, vb{};
#if defined(__AVX__)
            va.data = _mm_loadu_ps(a + i);
            vb.data = _mm_loadu_ps(b + i);
#elif defined(__ARM_NEON)
            va.data = vld1q_f32(a + i);
            vb.data = vld1q_f32(b + i);
#else
            for (int j = 0; j < 4; j++) {
                va.data[j] = a[i + j];
                vb.data[j] = b[i + j];
            }
#endif
            Vec4f vc = vadd(va, vb);

#if defined(__AVX__)
            _mm_storeu_ps(out + i, vc.data);
#elif defined(__ARM_NEON)
            vst1q_f32(out + i, vc.data);
#else
            for (int j = 0; j < 4; j++) {
                out[i + j] = vc.data[j];
            }
#endif
        }

        // scalar remainder
        for (; i < n; i++) {
            out[i] = a[i] + b[i];
        }
    }


    inline void sub_arrays(const f32* a, const f32* b, f32* out, size_t n) {
        size_t i = 0;

        // process 4 elements at a time
        for (; i + 4 <= n; i += 4) {
            Vec4f va{}, vb{};
#if defined(__AVX__)
            va.data = _mm_loadu_ps(a + i);
            vb.data = _mm_loadu_ps(b + i);
#elif defined(__ARM_NEON)
            va.data = vld1q_f32(a + i);
            vb.data = vld1q_f32(b + i);
#else
            for (int j = 0; j < 4; j++) {
                va.data[j] = a[i + j];
                vb.data[j] = b[i + j];
            }
#endif
            Vec4f vc = vsub(va, vb);

#if defined(__AVX__)
            _mm_storeu_ps(out + i, vc.data);
#elif defined(__ARM_NEON)
            vst1q_f32(out + i, vc.data);
#else
            for (int j = 0; j < 4; j++) {
                out[i + j] = vc.data[j];
            }
#endif
        }

        // scalar remainder
        for (; i < n; i++) {
            out[i] = a[i] - b[i];
        }
    }

    inline void mul_arrays(const f32* a, const f32* b, f32* out, size_t n) {
        size_t i = 0;

        // process 4 elements at a time
        for (; i + 4 <= n; i += 4) {
            Vec4f va{}, vb{};
#if defined(__AVX__)
            va.data = _mm_loadu_ps(a + i);
            vb.data = _mm_loadu_ps(b + i);
#elif defined(__ARM_NEON)
            va.data = vld1q_f32(a + i);
            vb.data = vld1q_f32(b + i);
#else
            for (int j = 0; j < 4; j++) {
                va.data[j] = a[i + j];
                vb.data[j] = b[i + j];
            }
#endif
            Vec4f vc = vmul(va, vb);

#if defined(__AVX__)
            _mm_storeu_ps(out + i, vc.data);
#elif defined(__ARM_NEON)
            vst1q_f32(out + i, vc.data);
#else
            for (int j = 0; j < 4; j++) {
                out[i + j] = vc.data[j];
            }
#endif
        }

        // scalar remainder
        for (; i < n; i++) {
            out[i] = a[i] * b[i];
        }
    }

    inline void div_arrays(const f32* a, const f32* b, f32* out, size_t n) {
        size_t i = 0;

        // process 4 elements at a time
        for (; i + 4 <= n; i += 4) {
            Vec4f va{}, vb{};
#if defined(__AVX__)
            va.data = _mm_loadu_ps(a + i);
            vb.data = _mm_loadu_ps(b + i);
#elif defined(__ARM_NEON)
            va.data = vld1q_f32(a + i);
            vb.data = vld1q_f32(b + i);
#else
            for (int j = 0; j < 4; j++) {
                va.data[j] = a[i + j];
                vb.data[j] = b[i + j];
            }
#endif
            Vec4f vc = vdiv(va, vb);

#if defined(__AVX__)
            _mm_storeu_ps(out + i, vc.data);
#elif defined(__ARM_NEON)
            vst1q_f32(out + i, vc.data);
#else
            for (int j = 0; j < 4; j++) {
                out[i + j] = vc.data[j];
            }
#endif
        }

        // scalar remainder
        for (; i < n; i++) {
            out[i] = a[i] / b[i];
        }
    }


} // namespace hahaha::common::util
#endif // HAHAHA_VECTORIZE_H
