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

#include "Error.h"

#include <utility>

namespace hahaha::core
{
BaseError::BaseError(String msg) : msg_(std::move(msg))
{
}
BaseError::BaseError(String msg, String loc)
    : msg_(std::move(msg)), loc_(std::move(loc))
{
}
BaseError::BaseError(const char* msg) : msg_(String(msg))
{
}
BaseError::BaseError(const char* msg, const char* loc)
    : msg_(String(msg)), loc_(String(loc))
{
}
BaseError::BaseError() : msg_(String("")), loc_(String(""))
{
}
String BaseError::typeName() const
{
    return TypeName;
}
String BaseError::message() const
{
    return msg_;
}
String BaseError::location() const
{
    return loc_;
}
String BaseError::toString() const
{
    return String("Error: ") + msg_ + " at " + loc_;
}
} // namespace hahaha::core
