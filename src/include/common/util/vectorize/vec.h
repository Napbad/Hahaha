#ifndef HIAHIAHIA_VEC_H
#define HIAHIAHIA_VEC_H

#include <cstddef>
#include <type_traits>
#include <array>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace hiahiahia {
namespace vectorize {

template<typename T, size_t N>
class Vec {
    // Ensure T is an arithmetic type (int, long, etc.)
    static_assert(std::is_arithmetic_v<T>, "Vec only supports arithmetic types");
    static_assert(N > 0, "Vector size must be positive");

public:
    using value_type = T;
    using size_type = size_t;
    using reference = T&;
    using const_reference = const T&;
    using iterator = typename std::array<T, N>::iterator;
    using const_iterator = typename std::array<T, N>::const_iterator;
    static constexpr size_type size = N;

    // Default constructor - zero initialization
    Vec() noexcept : data_{} {}

    // Virtual destructor for safe inheritance
    virtual ~Vec() = default;

    // Constructor from scalar
    explicit Vec(T scalar) noexcept {
        std::fill_n(data_.begin(), N, scalar);
    }

    // Constructor from array
    explicit Vec(const std::array<T, N>& arr) noexcept : data_(arr) {}

    // Constructor from initializer list
    template<typename... Args,
             typename = std::enable_if_t<sizeof...(Args) == N &&
                                       (std::is_convertible_v<Args, T> && ...)>>
    explicit Vec(Args... args) noexcept : data_{static_cast<T>(args)...} {}

    // Copy constructor
    Vec(const Vec&) noexcept = default;

    // Move constructor
    Vec(Vec&&) noexcept = default;

    // Copy assignment
    Vec& operator=(const Vec&) noexcept = default;

    // Move assignment
    Vec& operator=(Vec&&) noexcept = default;

    // Array access
    reference operator[](size_type idx) {
        if (idx >= N) throw std::out_of_range("Vec index out of range");
        return data_[idx];
    }

    const_reference operator[](size_type idx) const {
        if (idx >= N) throw std::out_of_range("Vec index out of range");
        return data_[idx];
    }

    // Arithmetic operations
    Vec operator+(const Vec& rhs) const noexcept {
        Vec result;
        for (size_type i = 0; i < N; ++i) {
            result.data_[i] = data_[i] + rhs.data_[i];
        }
        return result;
    }

    Vec operator-(const Vec& rhs) const noexcept {
        Vec result;
        for (size_type i = 0; i < N; ++i) {
            result.data_[i] = data_[i] - rhs.data_[i];
        }
        return result;
    }

    Vec operator*(const Vec& rhs) const noexcept {
        Vec result;
        for (size_type i = 0; i < N; ++i) {
            result.data_[i] = data_[i] * rhs.data_[i];
        }
        return result;
    }

    Vec operator/(const Vec& rhs) const {
        Vec result;
        for (size_type i = 0; i < N; ++i) {
            if (rhs.data_[i] == T(0)) throw std::domain_error("Division by zero");
            result.data_[i] = data_[i] / rhs.data_[i];
        }
        return result;
    }

    // Scalar operations
    Vec operator+(T scalar) const noexcept {
        Vec result;
        for (size_type i = 0; i < N; ++i) {
            result.data_[i] = data_[i] + scalar;
        }
        return result;
    }

    Vec operator-(T scalar) const noexcept {
        Vec result;
        for (size_type i = 0; i < N; ++i) {
            result.data_[i] = data_[i] - scalar;
        }
        return result;
    }

    Vec operator*(T scalar) const noexcept {
        Vec result;
        for (size_type i = 0; i < N; ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    Vec operator/(T scalar) const {
        if (scalar == T(0)) throw std::domain_error("Division by zero");
        Vec result;
        for (size_type i = 0; i < N; ++i) {
            result.data_[i] = data_[i] / scalar;
        }
        return result;
    }

    // Compound assignment operators
    Vec& operator+=(const Vec& rhs) noexcept {
        for (size_type i = 0; i < N; ++i) {
            data_[i] += rhs.data_[i];
        }
        return *this;
    }

    Vec& operator-=(const Vec& rhs) noexcept {
        for (size_type i = 0; i < N; ++i) {
            data_[i] -= rhs.data_[i];
        }
        return *this;
    }

    Vec& operator*=(const Vec& rhs) noexcept {
        for (size_type i = 0; i < N; ++i) {
            data_[i] *= rhs.data_[i];
        }
        return *this;
    }

    Vec& operator/=(const Vec& rhs) {
        for (size_type i = 0; i < N; ++i) {
            if (rhs.data_[i] == T(0)) throw std::domain_error("Division by zero");
            data_[i] /= rhs.data_[i];
        }
        return *this;
    }

    // Scalar compound assignment
    Vec& operator+=(T scalar) noexcept {
        for (auto& val : data_) val += scalar;
        return *this;
    }

    Vec& operator-=(T scalar) noexcept {
        for (auto& val : data_) val -= scalar;
        return *this;
    }

    Vec& operator*=(T scalar) noexcept {
        for (auto& val : data_) val *= scalar;
        return *this;
    }

    Vec& operator/=(T scalar) {
        if (scalar == T(0)) throw std::domain_error("Division by zero");
        for (auto& val : data_) val /= scalar;
        return *this;
    }

    // Comparison operators
    bool operator==(const Vec& rhs) const noexcept {
        return data_ == rhs.data_;
    }

    bool operator!=(const Vec& rhs) const noexcept {
        return !(*this == rhs);
    }

    // Utility functions
    T sum() const noexcept {
        return std::accumulate(data_.begin(), data_.end(), T(0));
    }

    T product() const noexcept {
        return std::accumulate(data_.begin(), data_.end(), T(1), std::multiplies<T>());
    }

    T min() const noexcept {
        return *std::min_element(data_.begin(), data_.end());
    }

    T max() const noexcept {
        return *std::max_element(data_.begin(), data_.end());
    }

    // Iterator support
    iterator begin() noexcept { return data_.begin(); }
    iterator end() noexcept { return data_.end(); }
    const_iterator begin() const noexcept { return data_.begin(); }
    const_iterator end() const noexcept { return data_.end(); }
    const_iterator cbegin() const noexcept { return data_.cbegin(); }
    const_iterator cend() const noexcept { return data_.cend(); }

private:
    std::array<T, N> data_;
};

// Helper functions for creating vectors
template<typename T, size_t N>
Vec<T, N> make_vec(const std::array<T, N>& arr) noexcept {
    return Vec<T, N>(arr);
}

template<typename T>
Vec<T, 1> make_vec(T scalar) noexcept {
    return Vec<T, 1>(scalar);
}

template<typename T, typename... Args,
         typename = std::enable_if_t<(std::is_convertible_v<Args, T> && ...)>>
auto make_vec(T first, Args... args) noexcept {
    return Vec<T, 1 + sizeof...(args)>(first, args...);
}

// Scalar operations (reverse order)
template<typename T, size_t N>
Vec<T, N> operator+(T scalar, const Vec<T, N>& vec) noexcept {
    return vec + scalar;
}

template<typename T, size_t N>
Vec<T, N> operator*(T scalar, const Vec<T, N>& vec) noexcept {
    return vec * scalar;
}

} // namespace vectorize
} // namespace hiahiahia

#endif // HIAHIAHIA_VEC_H 