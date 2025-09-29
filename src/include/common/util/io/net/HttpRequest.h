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
#include "common/ds/map.h"


namespace hahaha::common::util {
  class HttpRequest {
public:
    HttpRequest(const ds::Str &url, HttpMethod method);

    void addHeader(const ds::Str &key, const ds::Str &value);
    void setBody(const ds::Str &body);

    [[nodiscard]] const ds::Str &getUrl() const { return _url; }
    [[nodiscard]] HttpMethod getMethod() const { return _method; }
    [[nodiscard]] const ds::Map<ds::Str, ds::Str> &getHeaders() const { return _headers; }
    [[nodiscard]] const ds::Str &getBody() const { return _body; }

private:
    ds::Str _url;
    HttpMethod _method;
    ds::Map<ds::Str, ds::Str> _headers;
    ds::Str _body;
  };


} // namespace hahaha::common::util


#endif // HIAHIAHIA_HTTPREQUEST_H
