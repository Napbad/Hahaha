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
// Created by root on 7/27/25.
//

#ifndef RES_H
#define RES_H

#include <memory>
#include <optional>
#include <stdexcept>
#include <variant>

namespace hiahiahia {

  template<class T, class E = std::unique_ptr<Err>>
  class Res {
    static_assert(!std::is_same_v<T, E>, "T and E cannot be the same type");
    static_assert(std::is_move_constructible_v<T> && std::is_move_constructible_v<E>,
                  "T and E must be move constructible");

public:
    template<typename U = T>
    explicit Res(U &&value, std::enable_if_t<!std::is_same_v<std::decay_t<U>, Res>, int> = 0) :
        data(std::forward<U>(value)) {}


    explicit Res(const T &value) : data(value) {}
    explicit Res(T &&value) : data(std::move(value)) {}
    explicit Res(const E &error) : data(error) {}
    explicit Res(E &&error) : data(std::move(error)) {}

    Res(const Res &other) = default;
    Res(Res &&other) noexcept = default;
    Res &operator=(const Res &other) = default;
    Res &operator=(Res &&other) noexcept = default;

    [[nodiscard]] bool isOk() const { return std::holds_alternative<T>(data); }

    [[nodiscard]] bool isErr() const { return std::holds_alternative<E>(data); }

    T &unwrap() {
      if (isOk()) {
        return std::get<T>(data);
      }
      throw std::runtime_error("Called `res::unwrap()` on an `Err` value");
    }

    E &unwrapErr() {
      if (isErr()) {
        return std::get<E>(data);
      }
      throw std::runtime_error("Called `res::unwrapErr()` on an `Ok` value");
    }

    T &&unwrap() && {
      if (isOk()) {
        return std::move(std::get<T>(data));
      }
      throw std::runtime_error("Called `res::unwrap()` on an `Err` value");
    }

    E &unwrapErr() & {
      if (isErr()) {
        return std::get<E>(data);
      }
      throw std::runtime_error("Called `res::unwrapErr()` on an `Ok` value");
    }

    const E &unwrapErr() const & {
      if (isErr()) {
        return std::get<E>(data);
      }
      throw std::runtime_error("Called `res::unwrapErr()` on an `Ok` value");
    }

    E &&unwrapErr() && {
      if (isErr()) {
        return std::move(std::get<E>(data));
      }
      throw std::runtime_error("Called `res::unwrapErr()` on an `Ok` value");
    }

    std::optional<T> ok() & {
      if (isOk()) {
        return std::get<T>(data);
      }
      return std::nullopt;
    }

    std::optional<T> ok() const & {
      if (isOk()) {
        return std::get<T>(data);
      }
      return std::nullopt;
    }

    std::optional<T> ok() && {
      if (isOk()) {
        return std::move(std::get<T>(data));
      }
      return std::nullopt;
    }

    std::optional<E> err() & {
      if (isErr()) {
        return std::get<E>(data);
      }
      return std::nullopt;
    }

    std::optional<E> err() const & {
      if (isErr()) {
        return std::get<E>(data);
      }
      return std::nullopt;
    }

    std::optional<E> err() && {
      if (isErr()) {
        return std::move(std::get<E>(data));
      }
      return std::nullopt;
    }

    T unwrapOr(T default_value) {
      if (isOk()) {
        return std::get<T>(data);
      }
      return default_value;
    }

    T unwrapOr(T default_value) const & {
      if (isOk()) {
        return std::get<T>(data);
      }
      return default_value;
    }

    T unwrapOr(T default_value) && {
      if (isOk()) {
        return std::move(std::get<T>(data));
      }
      return default_value;
    }

    template<typename F>
    T unwrapOrElse(F &&f) {
      if (isOk()) {
        return std::get<T>(data);
      }
      return std::forward<F>(f)(std::get<E>(data));
    }

    template<typename F>
    auto andThen(F &&f) & {
      using ReturnType = std::invoke_result_t<F, T &>;
      static_assert(is_res_v<ReturnType>, "F must return a Res type to support chain call");
      if (isOk()) {
        return std::forward<F>(f)(std::get<T>(data));
      }
      return ReturnType(std::get<E>(data));
    }


    template<typename F>
    auto andThen(F &&f) const & {
      using ReturnType = std::invoke_result_t<F, const T &>;
      static_assert(is_res_v<ReturnType>, "F must return a Res type to support chain call");
      if (isOk()) {
        return std::forward<F>(f)(std::get<T>(data));
      }
      return ReturnType(std::get<E>(data));
    }

    template<typename F>
    auto andThen(F &&f) && {
      using ReturnType = std::invoke_result_t<F, T &&>;
      static_assert(is_res_v<ReturnType>, "F must return a Res type to support chain call");
      if (isOk()) {
        return std::forward<F>(f)(std::move(std::get<T>(data)));
      }
      return ReturnType(std::move(std::get<E>(data)));
    }

    template<typename F>
    auto orElse(F &&f) & {
      using ReturnType = std::invoke_result_t<F, E &>;
      static_assert(is_res_v<ReturnType>, "F must return a Res type to support chain call");

      if (isErr()) {
        return std::forward<F>(f)(std::get<E>(data));
      }
      return ReturnType(std::get<T>(data));
    }

    template<typename F>
    auto orElse(F &&f) const & {
      using ReturnType = std::invoke_result_t<F, const E &>;
      static_assert(is_res_v<ReturnType>, "F must return a Res type to support chain call");

      if (isErr()) {
        return std::forward<F>(f)(std::get<E>(data));
      }
      return ReturnType(std::get<T>(data));
    }


    template<typename F>
    auto orElse(F &&f) && {
      using ReturnType = std::invoke_result_t<F, E &&>;
      static_assert(is_res_v<ReturnType>, "F must return a Res type to support chain call");

      if (isErr()) {
        return std::forward<F>(f)(std::move(std::get<E>(data)));
      }
      return ReturnType(std::move(std::get<T>(data)));
    }

    template<typename U, typename G>
    bool operator==(const Res<U, G> &other) const {
      if (isOk() && other.isOk()) {
        return std::get<T>(data) == std::get<U>(other.data);
      }
      if (isErr() && other.isErr()) {
        return std::get<E>(data) == std::get<G>(other.data);
      }
      return false;
    }

    template<typename U, typename G>
    bool operator!=(const Res<U, G> &other) const {
      return !(*this == other);
    }

    static Res Ok(const T &value) { return Res(value); }
    static Res Ok(T &&value) { return Res(std::move(value)); }
    static Res Err(const E &error) { return Res(error); }
    static Res Err(E &&error) { return Res(std::move(error)); }
private:
    std::variant<T, E> data;

    template<typename R>
    struct is_res : std::false_type {};

    template<typename U, typename G>
    struct is_res<Res<U, G>> : std::true_type {};

    template<typename R>
    static constexpr bool is_res_v = is_res<R>::value;
  };

} // namespace hiahiahia

#endif // RES_H
