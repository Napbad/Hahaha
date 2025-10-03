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
// Created by Napbad on 8/4/25.
//

#ifndef COMMONUTIL_H
#define COMMONUTIL_H
#include "common/Res.h"
#include "common/error.h"
#include <functional>
#include <utility>

namespace hahaha::common::ds {
    class Str;
}
namespace hahaha::common::util {

    class ConvertErr final : public error {

    public:
        explicit ConvertErr(const ds::Str& msg, ds::Str loca = ds::Str("Unknown")) : _location(std::move(loca)) {
            _message = "ConvertErr: " + msg;
        }
        ConvertErr(const char* str, const char* text) {
            _message  = ds::Str(str);
            _location = ds::Str(text);
        }
        ConvertErr(const ds::Str& str, const char* text) {
            _message  = ds::Str(str);
            _location = ds::Str(text);
        }
        ~ConvertErr() override = default;

        // Get the error type name
        [[nodiscard]] ds::Str typeName() const override {
            return ds::Str("ConvertErr");
        }

        // Get the error message
        [[nodiscard]] ds::Str message() const override {
            return _message;
        }

        // Get the error location
        [[nodiscard]] ds::Str location() const override {
            return _location;
        }

        // Convert the error to a string
        [[nodiscard]] ds::Str toString() const override {
            return _message + " at: " + _location;
        }

    private:
        ds::Str _message;
        ds::Str _location;
    };

    template <typename Target>
    Res<Target, ConvertErr> strTo(const ds::Str&) {
        return Res<Target, std::unique_ptr<ConvertErr>>();
    }

    /**
     * Helper function to trim whitespace from both ends of a string
     * @param s Input string to trim
     * @return Trimmed string
     */
    inline ds::Str trim(const ds::Str& s) {
        auto start = s.begin();
        while (start != s.end() && std::isspace(*start)) {
            start++;
        }

        auto end = s.end();
        do {
            end--;
        } while (std::distance(start, end) > 0 && std::isspace(*end));

        return {start, end + 1};
    }
    /**
     * Specialization for int type conversion
     * @param str Input string to convert
     * @return Result containing either int value or ConvertErr
     */
    template <>
    inline auto strTo<int>(const ds::Str& str) -> Res<int, ConvertErr> {
        SetRetT(int, ConvertErr) try {
            const ds::Str trimmed = trim(str);
            size_t pos;
            const int value = std::stoi(trimmed.c_str(), &pos);
            // Check if the entire string was consumed
            if (pos != trimmed.length()) {
                return RetType::err(newE(ConvertErr, "Invalid integer format", __func__));
            }
            Ok(value);
        } catch (const std::exception& e) {
            return RetType::err(ConvertErr(e.what(), __func__));
            Err(ConvertErr(e.what(), __func__));
        }
    }

    /**
     * Specialization for double type conversion
     * @param str Input string to convert
     * @return Result containing either double value or ConvertErr
     */
    template <>
    inline auto strTo<double>(const ds::Str& str) -> Res<double, ConvertErr> {
        SetRetT(double, ConvertErr) try {
            const ds::Str trimmed = trim(str);
            size_t pos;
            const double value = std::stod(trimmed.c_str(), &pos);
            // Check if the entire string was consumed
            if (pos != trimmed.length()) {
                Err(ConvertErr("Invalid double format", __func__));
            }
            Ok(value);
        } catch (const std::exception& e) {
            Err(ConvertErr(e.what(), __func__));
        }
    }

    /**
     * Specialization for f32 type conversion
     * @param str Input string to convert
     * @return Result containing either f32 value or ConvertErr
     */
    template <>
    inline auto strTo<f32>(const ds::Str& str) -> Res<f32, ConvertErr> {
        SetRetT(f32, ConvertErr) try {
            const ds::Str trimmed = trim(str);
            size_t pos;
            const f32 value = std::stof(trimmed.c_str(), &pos);
            // Check if the entire string was consumed
            if (pos != trimmed.length()) {
                Err(ConvertErr("Invalid f32 format", __func__));
            }
            Ok(value);
        } catch (const std::exception& e) {
            Err(ConvertErr(e.what(), __func__));
        }
    }
} // namespace hahaha::common::util


#endif // COMMONUTIL_H
