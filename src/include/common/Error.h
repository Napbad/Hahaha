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

#ifndef ERR_H
#define ERR_H
#include "ds/str.h"

namespace hiahiahia {
    class Error {
    public:
      virtual ~Error() = default;

      // Get the error type name
      [[nodiscard]] virtual ds::Str typeName() const = 0;

      // Get the error message
      [[nodiscard]] virtual ds::Str message() const = 0;

      // Get the error location
      [[nodiscard]] virtual ds::Str location() const = 0;

      // Convert the error to a string
      [[nodiscard]] virtual ds::Str toString() const = 0;
    };

} // namespace hiahiahia


#endif //ERR_H
