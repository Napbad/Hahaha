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
// Created by Napbad on 9/8/25.
//

#ifndef HIAHIAHIA_FILEUTIL_H
#define HIAHIAHIA_FILEUTIL_H

#include <common/ds/str.h>
#include <common/Res.h>
#include <common/Error.h>

namespace hahaha::common::util {

/**
 * File error class
 */
class FileError final : public Error {
public:
  explicit FileError(const ds::Str& message, const ds::Str& location = ds::Str("FileUtil")) :
    _message(message), _location(location) {}

  explicit FileError(const char* message, const ds::Str& location = ds::Str("FileUtil")) :
    _message(ds::Str(message)), _location(location) {}

  [[nodiscard]] ds::Str typeName() const override { return ds::Str("FileError"); }
  [[nodiscard]] ds::Str message() const override { return _message; }
  [[nodiscard]] ds::Str location() const override { return _location; }
  [[nodiscard]] ds::Str toString() const override {
    return typeName() + ds::Str(": ") + message() + ds::Str(" at ") + location();
  }

private:
  ds::Str _message;
  ds::Str _location;
};

/**
 * Read a file into a string
 */
Res<ds::Str, FileError> readFile(const ds::Str& path);

} // namespace hahaha::common::util

#endif // HIAHIAHIA_FILEUTIL_H
