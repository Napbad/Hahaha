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

//
// Created by napbad on 3/26/26.
//

#ifndef HAHAHA_ERROR_H_FEDE53AD90D34788ACC93B3F7C3582C8
#define HAHAHA_ERROR_H_FEDE53AD90D34788ACC93B3F7C3582C8
#include <format>
#include <string>
#include <utility>

namespace h3::core {

enum class ErrorCode {
    BaseError = 1,
    MemoryError,
    DeviceNotSupportedError,
    DeviceNotAvailableError,
    InvalidArgument,
    RuntimeError,
};

class Error {
  public:
    Error(std::string message, const ErrorCode code)
        : m_message(std::move(message)), m_code(code) {
    }

    explicit Error(const char* str) : m_message(str), m_code(ErrorCode::BaseError) {
    }

    template <typename... Args>
    Error(const ErrorCode code,
          std::format_string<Args...> formatMsg,
          Args&&... args)
        : m_message(std::format(formatMsg, std::forward<Args>(args)...)),
          m_code(code){

          }
    explicit Error(const std::string& string) {
        m_message = string;
        m_code = ErrorCode::BaseError;
    }
    ;

    [[nodiscard]] std::string message() const {
        return m_message;
    }

    [[nodiscard]] ErrorCode code() const {
        return m_code;
    }

  private:
    std::string m_message;
    ErrorCode m_code;
};

template <typename... Args>
Error err(const ErrorCode code,
          std::format_string<Args...> fmt,
          Args&&... formatContent) {
    return Error(std::format(fmt, std::forward<Args>(formatContent)...), code);
}
} // namespace h3::core

#endif // HAHAHA_ERROR_H_FEDE53AD90D34788ACC93B3F7C3582C8