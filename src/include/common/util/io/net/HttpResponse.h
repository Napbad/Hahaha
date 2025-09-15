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

#ifndef HIAHIAHIA_HTTPRESPONSE_H
#define HIAHIAHIA_HTTPRESPONSE_H
#include "HttpMethod.h"
#include "common/ds/map.h"

namespace hiahiahia::util {

  class HttpResponse {
  public:
    HttpResponse() : _statusCode(0) {}

    void setStatusCode(const int code) { _statusCode = code; }
    void setHeader(const ds::Str& key, const ds::Str& value);
    void setBody(const ds::Str& body) { _body = body; }

    [[nodiscard]] int getStatusCode() const { return _statusCode; }
    [[nodiscard]] const ds::Map<ds::Str, ds::Str>& getHeaders() const { return _headers; }
    [[nodiscard]] const ds::Str& getBody() const { return _body; }

  private:
    int _statusCode;
    ds::Map<ds::Str, ds::Str> _headers;
    ds::Str _body;
  };

}
#endif // HIAHIAHIA_HTTPRESPONSE_H
