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
// Created by napbad on 9/8/25.
//

#include "io/net/HttpRequest.h"
#include <curl/curl.h>

#include "ds/String.h"
#include "io/net/HttpResponse.h"

namespace hahaha::core::util
{

HttpRequest::HttpRequest(const ds::String& url, const HttpMethod method) : url_(url), method_(method)
{
}

void HttpRequest::addHeader(const ds::String& key, const ds::String& value)
{
    headers_[key] = value;
}

void HttpRequest::setBody(const ds::String& body)
{
    body_ = body;
}
} // namespace hahaha::core::util
