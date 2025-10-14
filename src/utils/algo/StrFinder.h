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
// Created by Napbad on 8/1/25.
//

#ifndef STRFINDER_H
#define STRFINDER_H
#include <memory>

#include "core/defines/h3defs.h"
#include "core/ds/String.h"

using hahaha::core::ds::String;

namespace hahaha::core::ds
{

// KMP
class StrFinder
{

  public:
    explicit StrFinder(String& str, bool lateinit = false)
    {
        _str = &str;
        _len = str.size();
        _next = nullptr;

        if (!lateinit)
        {
            buildNextVal();
        }
    }

    ~StrFinder()
    {
        delete[] _next;
    }

    [[nodiscard]] sizeT find(char c) const
    {
        for (sizeT i = 0; i < _len; ++i)
        {
            if ((*_str)[i] == c)
            {
                return i;
            }
        }
        return String::npos;
    }

    sizeT find(String& str)
    {

        sizeT i = 0, j = 0;
        while (j < _len)
        {
            if ((*_str)[j] == str[i])
            {
                ++i;
                if (i == str.size())
                {
                    return j - i + 1;
                }
                ++j;
            }
            else
            {
                i = _next[i];
            }
        }

        return String::npos;
    }

  private:
    String* _str;
    int* _next;
    sizeT _len;

    void buildNextVal()
    {
        _next = new int[_len];
        _next[0] = -1;

        sizeT i = -1, j = 0;
        while (j < _len - 1)
        {
            if (i == -1 || (*_str)[i] == (*_str)[j])
            {
                ++i;
                ++j;
                _next[j] = static_cast<int>(i);
            }
            else
            {
                i = _next[i];
            }
        }
    }
};

} // namespace hahaha::core::ds

#endif // STRFINDER_H
