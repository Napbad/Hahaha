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

#include "io/file/fileUtil.h"

#include <filesystem>
#include <fstream>
#include <sstream>

#include "Res.h"

namespace hahaha::util::io
{

using namespace hahaha::common;

void deleteFile(const ds::String& path)
{
    std::filesystem::path p((path.data()));
    std::filesystem::remove(p);
}

bool fileExists(const ds::String& path)
{
    std::filesystem::path p((path.data()));
    return std::filesystem::exists(p);
}
bool createDir(const ds::String& dir)
{
    std::filesystem::path p((dir.data()));
    return std::filesystem::create_directory(p);
}
hahaha::Res<ds::String, FileError> readFile(const ds::String& path)
{
    SetRetT(ds::String, FileError);

    std::ifstream file(path.data());
    if (!file)
    {
        Err(FileError("File not found"));
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    Ok(ds::String(buffer.str()));
}
bool writeFile(const ds::String& path, const ds::String& content)
{
    std::ofstream file(path.data());
    if (!file.is_open())
    {
        return false;
    }

    file << content.data();
    return file.good();
}
} // namespace hahaha::util::io
