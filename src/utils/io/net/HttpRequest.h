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

#ifndef HIAHIAHIA_HTTPREQUEST_H
#define HIAHIAHIA_HTTPREQUEST_H

#include "HttpMethod.h"
#include "ds/map.h"

namespace hahaha::common::util
{
class HttpRequest
{
  public:
    HttpRequest(const ds::String& url, HttpMethod method);

    void addHeader(const ds::String& key, const ds::String& value);
    void setBody(const ds::String& body);

    [[nodiscard]] const ds::String& getUrl() const
    {
        return url_;
    }
    [[nodiscard]] HttpMethod getMethod() const
    {
        return method_;
    }
    [[nodiscard]] const ds::Map<ds::String, ds::String>& getHeaders() const
    {
        return headers_;
    }
    [[nodiscard]] const ds::String& getBody() const
    {
        return body_;
    }

  private:
    ds::String url_;
    HttpMethod method_;
    ds::Map<ds::String, ds::String> headers_;
    ds::String body_;
};

} // namespace hahaha::common::util

#endif // HIAHIAHIA_HTTPREQUEST_H
