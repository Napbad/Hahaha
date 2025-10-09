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
// Created by Napbad on 8/26/25.
//

#ifndef HIAHIAHIA_HTTPMETHOD_H
#define HIAHIAHIA_HTTPMETHOD_H

#include <string>

#include "defines/h3defs.h"
#include "ds/String.h"

namespace hahaha::common::util
{

/**
 * HTTP method enum
 */
enum class HttpMethod
{
    GET,
    POST,
    PUT,
    DELETE,
    HEAD,
    PATCH,
    OPTIONS
};

/**
 * Convert HttpMethod to string
 */
inline ds::String methodToString(const HttpMethod method)
{
    switch (method)
    {
    case HttpMethod::GET:
        return ds::String("GET");
    case HttpMethod::POST:
        return ds::String("POST");
    case HttpMethod::PUT:
        return ds::String("PUT");
    case HttpMethod::DELETE:
        return ds::String("DELETE");
    case HttpMethod::HEAD:
        return ds::String("HEAD");
    case HttpMethod::PATCH:
        return ds::String("PATCH");
    case HttpMethod::OPTIONS:
        return ds::String("OPTIONS");
    default:
        return ds::String("UNKNOWN");
    }
}
} // namespace hahaha::common::util

#endif // HIAHIAHIA_HTTPMETHOD_H
