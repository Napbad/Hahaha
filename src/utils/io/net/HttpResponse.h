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
#include "ds/String.h"
#include "ds/map.h"

namespace hahaha::core::util
{

class HttpResponse
{
  public:
    HttpResponse() : statusCode_(0)
    {
    }

    void setStatusCode(const int code)
    {
        statusCode_ = code;
    }
    void setHeader(const ds::String& key, const ds::String& value);
    void setBody(const ds::String& body)
    {
        body_ = body;
    }

    [[nodiscard]] int getStatusCode() const
    {
        return statusCode_;
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
    int statusCode_;
    ds::Map<ds::String, ds::String> headers_;
    ds::String body_;
};

} // namespace hahaha::core::util
#endif // HIAHIAHIA_HTTPRESPONSE_H
