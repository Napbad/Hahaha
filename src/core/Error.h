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

#ifndef ERROR_H
#define ERROR_H
#include "ds/String.h"

namespace hahaha::common
{
namespace ds
{
class String;
}
using ds::String;

class Error
{
  public:
    virtual ~Error() = default;

    // Get the error type name
    [[nodiscard]] virtual String typeName() const = 0;

    // Get the error message
    [[nodiscard]] virtual String message() const = 0;

    // Get the error location
    [[nodiscard]] virtual String location() const = 0;

    // Convert the error to a string
    [[nodiscard]] virtual String toString() const = 0;

    Error() = default;
};

class BaseError : public Error
{
  public:
    explicit BaseError(String msg);
    explicit BaseError(String msg, String loc);
    explicit BaseError(const char* msg);
    explicit BaseError(const char* msg, const char* loc);

    BaseError();
    ~BaseError() override = default;

    [[nodiscard]] String typeName() const override;
    [[nodiscard]] String message() const override;
    [[nodiscard]] String location() const override;
    [[nodiscard]] String toString() const override;

  private:
    String msg_;
    String loc_;
    const String TypeName = String("BaseError");
};

class IndexOutOfBoundError final : public BaseError
{
  public:
    explicit IndexOutOfBoundError(String msg) : BaseError(std::move(msg), String("IndexOutOfBoundError"))
    {
    }
    explicit IndexOutOfBoundError(String msg, String loc) : BaseError(std::move(msg), std::move(loc))
    {
    }
    explicit IndexOutOfBoundError(const char* msg) : BaseError(String(msg), String("IndexOutOfBoundError"))
    {
    }
    explicit IndexOutOfBoundError(const char* msg, const char* loc) : BaseError(String(msg), String(loc))
    {
    }

    [[nodiscard]] String typeName() const override
    {
        return String("IndexOutOfBoundError");
    }
};

// Add other common error types here as needed

} // namespace hahaha::common

#endif // ERROR_H
