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
// Created by root on 8/4/25.
//

#ifndef COMMONUTIL_H
#define COMMONUTIL_H
#include <utility>

#include "include/common/Err.h"
#include "include/common/Res.h"

namespace hiahiahia::ds {
  class Str;
}
namespace hiahiahia::util {

  class ConvertErr final : public Err {

public:
    explicit ConvertErr(const ds::Str &msg, ds::Str loca = ds::Str("Unknown")) : _location(std::move(loca)) {
      _message = "ConvertErr: " + msg;
    }

    ~ConvertErr() override = default;

    // Get the error type name
    [[nodiscard]] ds::Str typeName() const override { return ds::Str("ConvertErr"); };

    // Get the error message
    [[nodiscard]] ds::Str message() const override { return _message; };

    // Get the error location
    [[nodiscard]] ds::Str location() const override { return _location; };

    // Convert the error to a string
    [[nodiscard]] ds::Str toString() const override { return _message + " at: " + _location; };

private:
    ds::Str _message;
    ds::Str _location;
  };

  template<typename Target>
  Res<Target, ConvertErr> strTo(const ds::Str &str) {
    return Res<Target, ConvertErr>();
  }

      /**
     * Helper function to trim whitespace from both ends of a string
     * @param s Input string to trim
     * @return Trimmed string
     */
    inline ds::Str trim(const ds::Str &s) {
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
    template<>
    inline Res<int, ConvertErr> strTo<int>(const ds::Str &str) {
        try {
            ds::Str trimmed = trim(str);
            size_t pos;
            int value = std::stoi(trimmed, &pos);

            // Check if entire string was converted
            if (pos != trimmed.size()) {
                return ConvertErr("Contains invalid characters", "strTo<int>");
            }
            return Res(value);
        } catch (const std::invalid_argument &e) {
            return ConvertErr("Invalid argument format: " + ds::Str(e.what()), "strTo<int>");
        } catch (const std::out_of_range &e) {
            return ConvertErr("Value out of range: " + ds::Str(e.what()), "strTo<int>");
        } catch (const std::exception &e) {
            return ConvertErr("Conversion failed: " + ds::Str(e.what()), "strTo<int>");
        }
    }

    /**
     * Specialization for long type conversion
     * @param str Input string to convert
     * @return Result containing either long value or ConvertErr
     */
    template<>
    Res<long, ConvertErr> strTo<long>(const ds::Str &str) {
        try {
            ds::Str trimmed = trim(str);
            size_t pos;
            long value = std::stol(trimmed, &pos);

            if (pos != trimmed.size()) {
                return ConvertErr("Contains invalid characters", "strTo<long>");
            }
            return value;
        } catch (const std::invalid_argument &e) {
            return ConvertErr("Invalid argument format: " + ds::Str(e.what()), "strTo<long>");
        } catch (const std::out_of_range &e) {
            return ConvertErr("Value out of range: " + ds::Str(e.what()), "strTo<long>");
        } catch (const std::exception &e) {
            return ConvertErr("Conversion failed: " + ds::Str(e.what()), "strTo<long>");
        }
    }

    /**
     * Specialization for double type conversion
     * @param str Input string to convert
     * @return Result containing either double value or ConvertErr
     */
    template<>
    Res<double, ConvertErr> strTo<double>(const ds::Str &str) {
        try {
            ds::Str trimmed = trim(str);
            size_t pos;
            double value = std::stod(trimmed, &pos);

            if (pos != trimmed.size()) {
                return ConvertErr("Contains invalid characters", "strTo<double>");
            }
            return value;
        } catch (const std::invalid_argument &e) {
            return ConvertErr("Invalid argument format: " + ds::Str(e.what()), "strTo<double>");
        } catch (const std::out_of_range &e) {
            return ConvertErr("Value out of range: " + ds::Str(e.what()), "strTo<double>");
        } catch (const std::exception &e) {
            return ConvertErr("Conversion failed: " + ds::Str(e.what()), "strTo<double>");
        }
    }

    /**
     * Specialization for float type conversion
     * @param str Input string to convert
     * @return Result containing either float value or ConvertErr
     */
    template<>
    Res<float, ConvertErr> strTo<float>(const ds::Str &str) {
        try {
            ds::Str trimmed = trim(str);
            size_t pos;
            float value = std::stof(trimmed, &pos);

            if (pos != trimmed.size()) {
                return ConvertErr("Contains invalid characters", "strTo<float>");
            }
            return value;
        } catch (const std::invalid_argument &e) {
            return ConvertErr("Invalid argument format: " + ds::Str(e.what()), "strTo<float>");
        } catch (const std::out_of_range &e) {
            return ConvertErr("Value out of range: " + ds::Str(e.what()), "strTo<float>");
        } catch (const std::exception &e) {
            return ConvertErr("Conversion failed: " + ds::Str(e.what()), "strTo<float>");
        }
    }

    /**
     * Specialization for bool type conversion
     * Accepts "true", "false", "1", "0" (case-insensitive)
     * @param str Input string to convert
     * @return Result containing either bool value or ConvertErr
     */
    template<>
    Res<bool, ConvertErr> strTo<bool>(const ds::Str &str) {
        ds::Str trimmed = trim(str);
        // Convert to lowercase for case-insensitive comparison
        std::transform(trimmed.begin(), trimmed.end(), trimmed.begin(),
                      [](unsigned char c){ return std::tolower(c); });

        if (trimmed == "true" || trimmed == "1") {
            return true;
        } else if (trimmed == "false" || trimmed == "0") {
            return false;
        } else {
            return ConvertErr("Invalid boolean format, must be true/false or 1/0", "strTo<bool>");
        }
    }

    /**
     * Specialization for string type (returns trimmed string)
     * @param str Input string to process
     * @return Result containing trimmed string
     */
    template<>
    Res<ds::Str, ConvertErr> strTo<ds::Str>(const ds::Str &str) {
        return trim(str);
    }

} // namespace hiahiahia::util


#endif // COMMONUTIL_H
