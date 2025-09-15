#ifndef HIAHIAHIA_VECTORIZE_H
#define HIAHIAHIA_VECTORIZE_H

#include <cstdint>
#include <type_traits>
#include <array>

// Platform detection
#if defined(__x86_64__) || defined(_M_X64)
    #define HIAHIAHIA_X86_64 1
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define HIAHIAHIA_ARM64 1
    #include <arm_neon.h>
#endif

namespace hiahiahia {
namespace vectorize {

// Forward declarations
template<typename T, size_t N>
class alignas(32) VecRegister;

// SIMD operation tags for compile-time optimization
struct Add {};
struct Sub {};
struct Mul {};
struct Div {};

// Type traits for SIMD support
template<typename T>
struct SimdTraits {
    static constexpr bool is_supported = false;
    static constexpr size_t vector_size = 1;
};

// Specializations for common types
template<>
struct SimdTraits<float> {
    static constexpr bool is_supported = true;
    #if defined(HIAHIAHIA_X86_64)
        static constexpr size_t vector_size = 8; // AVX
    #elif defined(HIAHIAHIA_ARM64)
        static constexpr size_t vector_size = 4; // NEON
    #else
        static constexpr size_t vector_size = 1;
    #endif
};

template<>
struct SimdTraits<double> {
    static constexpr bool is_supported = true;
    #if defined(HIAHIAHIA_X86_64)
        static constexpr size_t vector_size = 4; // AVX
    #elif defined(HIAHIAHIA_ARM64)
        static constexpr size_t vector_size = 2; // NEON
    #else
        static constexpr size_t vector_size = 1;
    #endif
};

// Main vectorized register class
template<typename T, size_t N>
class VecRegister {
    static_assert(std::is_arithmetic_v<T>, "VecRegister only supports arithmetic types");
    static_assert(N > 0, "Vector size must be positive");

public:
    using value_type = T;
    static constexpr size_t size = N;

    // Default constructor - zero initialization
    VecRegister() : data_{} {}

    // Constructor from scalar
    explicit VecRegister(T scalar) {
        for (size_t i = 0; i < N; ++i) {
            data_[i] = scalar;
        }
    }

    // Constructor from array
    explicit VecRegister(const std::array<T, N>& arr) : data_(arr) {}

    // Array access
    T& operator[](size_t idx) { return data_[idx]; }
    const T& operator[](size_t idx) const { return data_[idx]; }

    // Arithmetic operations
    VecRegister operator+(const VecRegister& rhs) const {
        return arithmetic_op<Add>(rhs);
    }

    VecRegister operator-(const VecRegister& rhs) const {
        return arithmetic_op<Sub>(rhs);
    }

    VecRegister operator*(const VecRegister& rhs) const {
        return arithmetic_op<Mul>(rhs);
    }

    VecRegister operator/(const VecRegister& rhs) const {
        return arithmetic_op<Div>(rhs);
    }

private:
    template<typename Op>
    VecRegister arithmetic_op(const VecRegister& rhs) const {
        VecRegister result;
        if constexpr (SimdTraits<T>::is_supported) {
            #if defined(HIAHIAHIA_X86_64)
                if constexpr (std::is_same_v<T, float> && std::is_same_v<Op, Add>) {
                    for (size_t i = 0; i < N; i += 8) {
                        __m256 a = _mm256_load_ps(&data_[i]);
                        __m256 b = _mm256_load_ps(&rhs.data_[i]);
                        __m256 c = _mm256_add_ps(a, b);
                        _mm256_store_ps(&result.data_[i], c);
                    }
                }
                // Add more SIMD implementations for other types and operations
            #elif defined(HIAHIAHIA_ARM64)
                if constexpr (std::is_same_v<T, float> && std::is_same_v<Op, Add>) {
                    for (size_t i = 0; i < N; i += 4) {
                        float32x4_t a = vld1q_f32(&data_[i]);
                        float32x4_t b = vld1q_f32(&rhs.data_[i]);
                        float32x4_t c = vaddq_f32(a, b);
                        vst1q_f32(&result.data_[i], c);
                    }
                }
                // Add more SIMD implementations for other types and operations
            #endif
        } else {
            // Fallback scalar implementation
            for (size_t i = 0; i < N; ++i) {
                if constexpr (std::is_same_v<Op, Add>) {
                    result.data_[i] = data_[i] + rhs.data_[i];
                } else if constexpr (std::is_same_v<Op, Sub>) {
                    result.data_[i] = data_[i] - rhs.data_[i];
                } else if constexpr (std::is_same_v<Op, Mul>) {
                    result.data_[i] = data_[i] * rhs.data_[i];
                } else if constexpr (std::is_same_v<Op, Div>) {
                    result.data_[i] = data_[i] / rhs.data_[i];
                }
            }
        }
        return result;
    }

    alignas(32) std::array<T, N> data_;
};

// Helper function to create a vector register
template<typename T, size_t N>
VecRegister<T, N> make_vec(const std::array<T, N>& arr) {
    return VecRegister<T, N>(arr);
}

template<typename T>
VecRegister<T, SimdTraits<T>::vector_size> make_vec(T scalar) {
    return VecRegister<T, SimdTraits<T>::vector_size>(scalar);
}

} // namespace vectorize
} // namespace hiahiahia

#endif // HIAHIAHIA_VECTORIZE_H 