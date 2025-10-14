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

#include <ds/String.h>
#include <Error.h>
#include <Res.h>

namespace hahaha::util::io
{

/**
 * File error class
 */
class FileError final : public hahaha::core::Error
{
  public:
    explicit FileError(const core::ds::String& message,
                       const core::ds::String& location = core::ds::String("FileUtil"))
        : message_(message), location_(location)
    {
    }

    explicit FileError(const char* message, const core::ds::String& location = core::ds::String("FileUtil"))
        : message_(core::ds::String(message)), location_(location)
    {
    }

    [[nodiscard]] core::ds::String typeName() const override
    {
        return core::ds::String("FileError");
    }
    [[nodiscard]] core::ds::String message() const override
    {
        return message_;
    }
    [[nodiscard]] core::ds::String location() const override
    {
        return location_;
    }
    [[nodiscard]] core::ds::String toString() const override
    {
        return typeName() + core::ds::String(": ") + message() + core::ds::String(" at ") + location();
    }

  private:
    core::ds::String message_;
    core::ds::String location_;
};

/**
 * Read a file into a string
 */
hahaha::Res<hahaha::core::ds::String, FileError> readFile(const hahaha::core::ds::String& path);

void deleteFile(const hahaha::core::ds::String& path);

bool fileExists(const core::ds::String& path);

bool createDir(const core::ds::String& dir);

bool writeFile(const core::ds::String& path, const core::ds::String& content);

} // namespace hahaha::util::io

#endif // HIAHIAHIA_FILEUTIL_H
