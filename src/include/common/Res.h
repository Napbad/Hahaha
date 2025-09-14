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

#ifndef HIAHIAHIA_RES_H
#define HIAHIAHIA_RES_H

#include <memory>
#include <variant>
#include <type_traits>

namespace hiahiahia {

// Forward declaration for VoidType
struct VoidType {
  VoidType() = default;
};

// Helper type traits
template<typename T>
struct function_traits;

template<typename R, typename... Args>
struct function_traits<R(Args...)> {
  using return_type = R;
};

template<typename R, typename... Args>
struct function_traits<R(*)(Args...)> : function_traits<R(Args...)> {};

template<typename R, typename C, typename... Args>
struct function_traits<R(C::*)(Args...)> : function_traits<R(Args...)> {};

template<typename R, typename C, typename... Args>
struct function_traits<R(C::*)(Args...) const> : function_traits<R(Args...)> {};

template<typename T>
struct function_traits : function_traits<decltype(&T::operator())> {};

/**
 * Result type that can hold either a value or an error
 */
template<typename T, typename E>
class Res {
public:
  using ValueType = std::conditional_t<std::is_same_v<T, void>, VoidType, T>;
  using ErrorType = E;

  // Constructors for value
  template<typename U = T, typename = std::enable_if_t<!std::is_same_v<U, void>>>
  explicit Res(U&& value) : data(std::in_place_type<ValueType>, std::forward<U>(value)) {}

  template<typename U = T, typename = std::enable_if_t<!std::is_same_v<U, void>>>
  explicit Res(const U& value) : data(std::in_place_type<ValueType>, value) {}

  // Constructor for void
  template<typename U = T, typename = std::enable_if_t<std::is_same_v<U, void>>>
  explicit Res() : data(std::in_place_type<VoidType>) {}

  // Constructors for error
  explicit Res(E&& error) : data(std::in_place_type<std::unique_ptr<E>>, std::make_unique<E>(std::move(error))) {}
  explicit Res(const E& error) : data(std::in_place_type<std::unique_ptr<E>>, std::make_unique<E>(error)) {}
  explicit Res(std::unique_ptr<E>&& error) : data(std::in_place_type<std::unique_ptr<E>>, std::move(error)) {}
  explicit Res(E* error) : data(std::in_place_type<std::unique_ptr<E>>, std::unique_ptr<E>(error)) {}

  // Copy operations
  Res(const Res& other) {
    if (other.isOk()) {
      data = std::variant<ValueType, std::unique_ptr<E>>(std::in_place_index<0>, std::get<ValueType>(other.data));
    } else {
      data = std::variant<ValueType, std::unique_ptr<E>>(std::in_place_index<1>, std::make_unique<E>(*std::get<std::unique_ptr<E>>(other.data)));
    }
  }

  Res& operator=(const Res& other) {
    if (this != &other) {
      if (other.isOk()) {
        data = std::variant<ValueType, std::unique_ptr<E>>(std::in_place_index<0>, std::get<ValueType>(other.data));
      } else {
        data = std::variant<ValueType, std::unique_ptr<E>>(std::in_place_index<1>, std::make_unique<E>(*std::get<std::unique_ptr<E>>(other.data)));
      }
    }
    return *this;
  }

  // Move operations
  Res(Res&&) noexcept = default;
  Res& operator=(Res&&) noexcept = default;

  // Static factory methods for value
  template<typename U = T, typename = std::enable_if_t<!std::is_same_v<U, void>>>
  static Res ok(U&& value) { return Res(std::forward<U>(value)); }

  template<typename U = T, typename = std::enable_if_t<!std::is_same_v<U, void>>>
  static Res ok(const U& value) { return Res(value); }

  // Static factory method for void
  template<typename U = T, typename = std::enable_if_t<std::is_same_v<U, void>>>
  static Res ok() { return Res(); }

  // Static factory methods for error
  static Res err(E&& error) { return Res(std::forward<E>(error)); }
  static Res err(const E& error) { return Res(error); }
  static Res err(std::unique_ptr<E>&& error) { return Res(std::move(error)); }
  static Res err(E* error) { return Res(error); }

  // Status checks
  [[nodiscard]] bool isOk() const { return std::holds_alternative<ValueType>(data); }
  [[nodiscard]] bool isErr() const { return std::holds_alternative<std::unique_ptr<E>>(data); }

  // Value access
  template<typename U = T, typename = std::enable_if_t<!std::is_same_v<U, void>>>
  U& unwrap() & {
    if (!isOk()) throw std::runtime_error("Called unwrap() on an Err value");
    return std::get<ValueType>(data);
  }

  template<typename U = T, typename = std::enable_if_t<!std::is_same_v<U, void>>>
  const U& unwrap() const & {
    if (!isOk()) throw std::runtime_error("Called unwrap() on an Err value");
    return std::get<ValueType>(data);
  }

  template<typename U = T, typename = std::enable_if_t<!std::is_same_v<U, void>>>
  U&& unwrap() && {
    if (!isOk()) throw std::runtime_error("Called unwrap() on an Err value");
    return std::move(std::get<ValueType>(data));
  }

  // Error access
  E& unwrapErr() & {
    if (!isErr()) throw std::runtime_error("Called unwrapErr() on an Ok value");
    return *std::get<std::unique_ptr<E>>(data);
  }

  const E& unwrapErr() const & {
    if (!isErr()) throw std::runtime_error("Called unwrapErr() on an Ok value");
    return *std::get<std::unique_ptr<E>>(data);
  }

  E unwrapErr() && {
    if (!isErr()) throw std::runtime_error("Called unwrapErr() on an Ok value");
    return std::move(*std::get<std::unique_ptr<E>>(data));
  }

  // Map operations
  template<typename F>
  auto map(F&& f) const & {
    using RetType = typename function_traits<std::remove_reference_t<F>>::return_type;
    if (isOk()) {
      if constexpr (std::is_same_v<T, void>) {
        return Res<RetType, E>::ok(f());
      } else {
        return Res<RetType, E>::ok(f(std::get<ValueType>(data)));
      }
    }
    return Res<RetType, E>::err(E(*std::get<std::unique_ptr<E>>(data)));
  }

  template<typename F>
  auto map(F&& f) && {
    using RetType = typename function_traits<std::remove_reference_t<F>>::return_type;
    if (isOk()) {
      if constexpr (std::is_same_v<T, void>) {
        return Res<RetType, E>::ok(f());
      } else {
        return Res<RetType, E>::ok(f(std::move(std::get<ValueType>(data))));
      }
    }
    return Res<RetType, E>::err(std::move(*std::get<std::unique_ptr<E>>(data)));
  }

  // MapErr operations
  template<typename F>
  auto mapErr(F&& f) const & {
    using NewError = typename function_traits<std::remove_reference_t<F>>::return_type;
    if (isErr()) {
      return Res<T, NewError>::err(f(*std::get<std::unique_ptr<E>>(data)));
    }
    if constexpr (std::is_same_v<T, void>) {
      return Res<T, NewError>::ok();
    } else {
      return Res<T, NewError>::ok(std::get<ValueType>(data));
    }
  }

  template<typename F>
  auto mapErr(F&& f) && {
    using NewError = typename function_traits<std::remove_reference_t<F>>::return_type;
    if (isErr()) {
      return Res<T, NewError>::err(f(std::move(*std::get<std::unique_ptr<E>>(data))));
    }
    if constexpr (std::is_same_v<T, void>) {
      return Res<T, NewError>::ok();
    } else {
      return Res<T, NewError>::ok(std::move(std::get<ValueType>(data)));
    }
  }

private:
  std::variant<ValueType, std::unique_ptr<E>> data;
};

// Macros for easier use
#define SetRetT(T, E) using RetType = Res<T, E>;
#define Ok(Val) return RetType::ok(Val);
#define Err(Error) return RetType::err((Error));
#define newE(ErrorT, ...) (new ErrorT(__VA_ARGS__))

} // namespace hiahiahia

#endif // HIAHIAHIA_RES_H
