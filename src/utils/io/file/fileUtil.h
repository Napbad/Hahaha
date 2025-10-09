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
class FileError final : public hahaha::common::Error
{
  public:
    explicit FileError(const common::ds::String& message,
                       const common::ds::String& location = common::ds::String("FileUtil"))
        : message_(message), location_(location)
    {
    }

    explicit FileError(const char* message, const common::ds::String& location = common::ds::String("FileUtil"))
        : message_(common::ds::String(message)), location_(location)
    {
    }

    [[nodiscard]] common::ds::String typeName() const override
    {
        return common::ds::String("FileError");
    }
    [[nodiscard]] common::ds::String message() const override
    {
        return message_;
    }
    [[nodiscard]] common::ds::String location() const override
    {
        return location_;
    }
    [[nodiscard]] common::ds::String toString() const override
    {
        return typeName() + common::ds::String(": ") + message() + common::ds::String(" at ") + location();
    }

  private:
    common::ds::String message_;
    common::ds::String location_;
};

/**
 * Read a file into a string
 */
hahaha::Res<hahaha::common::ds::String, FileError> readFile(const hahaha::common::ds::String& path);

void deleteFile(const hahaha::common::ds::String& path);

bool fileExists(const common::ds::String& path);

bool createDir(const common::ds::String& dir);

bool writeFile(const common::ds::String& path, const common::ds::String& content);

} // namespace hahaha::util::io

#endif // HIAHIAHIA_FILEUTIL_H
