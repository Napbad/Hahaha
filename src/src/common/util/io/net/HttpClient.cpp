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
// Created by Napbad on 9/8/25.
//

#include <curl/curl.h>

#include "common/util/io/net/HttpClient.h"

#include "common/defines/h3defs.h"
#include "common/util/io/net/HttpRequest.h"

namespace hahaha::common::util {


  sizeT WriteCallback(void* dataBuffer, const sizeT blockSize, const sizeT blockCount, void* userData) {
    static_cast<ds::Str*>(userData)->append(static_cast<char*>(dataBuffer), blockSize * blockCount);
    return blockSize * blockCount;
  }

  std::unique_ptr<HttpResponse> HttpClient::send(const HttpRequest& request) {
    CURL* curl = curl_easy_init();
    auto response = std::make_unique<HttpResponse>();

    if (!curl) {
        return response; // Return empty response on error
    }

    try {
        // Set URL
        curl_easy_setopt(curl, CURLOPT_URL, request.getUrl().c_str());

        // Set HTTP method
        switch (request.getMethod()) {
            case HttpMethod::POST:
                curl_easy_setopt(curl, CURLOPT_POST, 1L);
                break;
            case HttpMethod::PUT:
                curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");
                break;
            case HttpMethod::DELETE:
                curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
                break;
            default:
                curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
        }

        // Set headers
        curl_slist * headers = nullptr;
        for (const auto &[fst, snd] : request.getHeaders()) {
            headers = curl_slist_append(headers,
                (fst + ": " + snd).c_str());
        }
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        // Set request body if present
        if (!request.getBody().empty()) {
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request.getBody().c_str());
        }

        // Capture response
        const ds::Str responseData;
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseData);

        // Execute request
        if (const CURLcode res = curl_easy_perform(curl); res == CURLE_OK) {
            long statusCode;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &statusCode);
            response->setStatusCode(static_cast<int>(statusCode));
            response->setBody(responseData);
        }

        // Cleanup
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    } catch (...) {
        curl_easy_cleanup(curl);
        throw;
    }

    return response;
}

} // namespace hahaha::common::ds
