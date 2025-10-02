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

#include "common/error.h"

#include <utility>

namespace hahaha::common {
    // BaseErr implementations
    BaseErr::BaseErr(ds::Str msg) : _msg(std::move(msg)) {}

    BaseErr::BaseErr(ds::Str msg, ds::Str loc) : _msg(std::move(msg)), _loc(std::move(loc)) {}

    BaseErr::BaseErr(const char* msg) : _msg(msg) {}

    BaseErr::BaseErr(const char* msg, const char* loc) : _msg(msg), _loc(loc) {}

    BaseErr::BaseErr() = default;

    ds::Str BaseErr::typeName() const {
        return TypeName;
    }

    ds::Str BaseErr::message() const {
        return _msg;
    }

    ds::Str BaseErr::location() const {
        return _loc;
    }

    ds::Str BaseErr::toString() const {
        return _msg;
    }
} // namespace hahaha::common