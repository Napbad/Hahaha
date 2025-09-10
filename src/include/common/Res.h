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

#ifndef RES_H
#define RES_H

#include <memory>
#include <stdexcept>
#include <variant>

#include "Error.h"
#include "Res.h"

namespace hiahiahia {


  /// Res<T, E> is a Rust-like Result type.
  /// - Holds either a value (T) or an error (E).
  /// - Errors are RAII-managed with std::unique_ptr.
  template<typename T, typename E>
  class Res {
public:
    using OkType = T;
    using ErrType = E;

private:
    std::variant<T, std::unique_ptr<E>> data;

    // ----------------------
    // Private constructors
    // ----------------------
    explicit Res(T &&value) : data(std::in_place_type<T>, std::move(value)) {}
    explicit Res(const T &value) : data(std::in_place_type<T>, value) {}

    explicit Res(E &&error) : data(std::in_place_type<std::unique_ptr<E>>, std::make_unique<E>(std::move(error))) {}
    explicit Res(const E &error) : data(std::in_place_type<std::unique_ptr<E>>, std::make_unique<E>(error)) {}
    explicit Res(std::unique_ptr<E> &&error) : data(std::in_place_type<std::unique_ptr<E>>, std::move(error)) {}
    explicit Res(E *error) : data(std::in_place_type<std::unique_ptr<E>>, std::unique_ptr<E>(error)) {}

public:
    // Copy / move
    Res(const Res &) = default;
    Res(Res &&) noexcept = default;
    Res &operator=(const Res &) = default;
    Res &operator=(Res &&) noexcept = default;

    // ----------------
    // Static factories
    // ----------------

    static Res ok(T &&value) { return Res(std::forward<T>(value)); }
    static Res ok(const T &value) { return Res(value); }

    static Res err(E &&error) { return Res(std::forward<E>(error)); }
    static Res err(const E &error) { return Res(error); }
    static Res err(std::unique_ptr<E> &&error) { return Res(std::move(error)); }
    static Res err(E *error) { return Res(error); }

    // ----------------
    // Queries
    // ----------------
    [[nodiscard]] bool isOk() const { return std::holds_alternative<T>(data); }
    [[nodiscard]] bool isErr() const { return std::holds_alternative<std::unique_ptr<E>>(data); }

    // ----------------
    // Unwrap (T)
    // ----------------
    T &unwrap() & {
      if (!isOk())
        throw std::runtime_error("unwrap() called on Err");
      return std::get<T>(data);
    }

    const T &unwrap() const & {
      if (!isOk())
        throw std::runtime_error("unwrap() called on Err");
      return std::get<T>(data);
    }

    T &&unwrap() && {
      if (!isOk())
        throw std::runtime_error("unwrap() called on Err");
      return std::move(std::get<T>(data));
    }

    // ----------------
    // UnwrapErr (E)
    // ----------------
    E &unwrapErr() & {
      if (!isErr())
        throw std::runtime_error("unwrapErr() called on Ok");
      return *std::get<std::unique_ptr<E>>(data);
    }

    const E &unwrapErr() const & {
      if (!isErr())
        throw std::runtime_error("unwrapErr() called on Ok");
      return *std::get<std::unique_ptr<E>>(data);
    }

    E unwrapErr() && {
      if (!isErr())
        throw std::runtime_error("unwrapErr() called on Ok");
      return std::move(*std::get<std::unique_ptr<E>>(data));
    }

    // ----------------
    // Functional helpers
    // ----------------
    template<typename F>
    auto map(F &&f) const & -> Res<std::invoke_result_t<F, const T &>, E> {
      using U = std::invoke_result_t<F, const T &>;
      if (isOk())
        return Res<U, E>::ok(f(std::get<T>(data)));
      return Res<U, E>::err(E(*std::get<std::unique_ptr<E>>(data)));
    }

    template<typename F>
    auto map(F &&f) && -> Res<std::invoke_result_t<F, T &&>, E> {
      using U = std::invoke_result_t<F, T &&>;
      if (isOk())
        return Res<U, E>::ok(f(std::move(std::get<T>(data))));
      return Res<U, E>::err(std::move(*std::get<std::unique_ptr<E>>(data)));
    }

    template<typename F>
    auto mapErr(F &&f) const & -> Res<T, std::invoke_result_t<F, const E &>> {
      using FErr = std::invoke_result_t<F, const E &>;
      if (isErr())
        return Res<T, FErr>::err(f(*std::get<std::unique_ptr<E>>(data)));
      return Res<T, FErr>::ok(std::get<T>(data));
    }

    template<typename F>
    auto mapErr(F &&f) && -> Res<T, std::invoke_result_t<F, E &&>> {
      using FErr = std::invoke_result_t<F, E &&>;
      if (isErr())
        return Res<T, FErr>::err(f(std::move(*std::get<std::unique_ptr<E>>(data))));
      return Res<T, FErr>::ok(std::move(std::get<T>(data)));
    }
  };


  /**
   * @brief Define return type alias for function
   *
   * This macro defines a type alias RetType pointing to Res<T, E> type
   * within function scope. It is typically used before Ok() or Err() macros
   * to set the return type.
   *
   * @param T The first template parameter of Res type, representing the type of value contained in success state
   * @param E The second template parameter of Res type, representing the error type contained in error state
   */
#define SetRetT(T, E) using RetType = Res<T, E>;

  /**
   * @brief Return a successful Res object
   *
   * This macro returns a successful state Res object containing the specified value.
   * Requires using SetRetT macro first to define RetType type alias.
   *
   * @param Val The value to be contained in the successful state Res object
   */
#define Ok(Val) return RetType::ok(Val);

  /**
   * @brief Return a failed Res object
   *
   * This macro returns a failed state Res object containing the specified error.
   * Requires using SetRetT macro first to define RetType type alias.
   *
   * @param Error The error to be contained in the failed state Res object
   */
#define Err(Error) return RetType::err((Error));

  /**
   * @brief Create a new error object
   *
   * This macro creates a new error object instance.
   * Requires using SetRetT macro first to define RetType type alias.
   *
   * @param ErrorT Error type
   * @param ... Arguments passed to the error constructor
   */
#define newE(ErrorT, ...) (new ErrorT(__VA_ARGS__))
} // namespace hiahiahia

#endif // RES_H
