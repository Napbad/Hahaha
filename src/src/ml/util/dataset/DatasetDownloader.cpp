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

#include <common/util/io/file/fileUtil.h>
#include <common/util/io/net/HttpClient.h> // Include HttpClient
#include <curl/curl.h>
#include <fstream>
#include <ml/util/dataset/DatasetDownloader.h>
#include <sstream>

namespace hahaha::ml {
    using namespace common;
    using namespace common::util;
    namespace {

        // This writeCallback will now write directly to the output stream provided
        size_t writeCallback(void* contents, const size_t size, size_t nmemb, void* userp) {
            auto* outStream           = static_cast<std::ofstream*>(userp);
            const size_t bytesToWrite = size * nmemb;
            outStream->write(static_cast<char*>(contents), static_cast<long>(bytesToWrite));
            return bytesToWrite;
        }

        // The progress callback can remain as is for now
        int progressCallback(const void* clientp, double dltotal, double dlnow, double ultotal, double ulnow) {
            (void) clientp;
            (void) dltotal;
            (void) dlnow;
            (void) ultotal;
            (void) ulnow;
            // You can add progress reporting logic here if showProgress is true
            return 0; // Return non-zero to abort transfer
        }

    } // namespace

    Res<void, DatasetDownloaderError> DatasetDownloader::downloadFromUrl(
        const Str& url, const Str& outputPath, const bool showProgress) {
        SetRetT(void, DatasetDownloaderError);

        // Use HttpRequest to prepare the request
        HttpRequest request(url, HttpMethod::GET);

        // Create an HttpClient and send the request
        std::unique_ptr<HttpResponse> response = HttpClient::send(request);

        if (!response) {
            Err(DatasetDownloaderError(Str("Failed to get HTTP response for URL: ") + url));
        }

        // Check HTTP status code
        if (response->getStatusCode() >= 400) {
            Err(DatasetDownloaderError(
                Str("HTTP error: ") + Str(std::to_string(response->getStatusCode())) + Str(" for URL: ") + url));
        }

        // Open output file
        std::ofstream outFile(outputPath.data(), std::ios::binary);
        if (!outFile.is_open()) {
            Err(DatasetDownloaderError(Str("Failed to open output file: ") + outputPath));
        }

        // Write the response body to the file
        outFile << response->getBody().c_str();
        outFile.close();

        Ok();
    }

    Res<void, DatasetDownloaderError> DatasetDownloader::downloadFromUCI(
        const Str& datasetName, const Str& outputPath) {
        const auto baseUrl = Str("https://archive.ics.uci.edu/ml/machine-learning-databases/");
        const Str url      = baseUrl + datasetName + Str("/") + datasetName + Str(".data");
        return downloadFromUrl(url, outputPath);
    }

    Res<void, DatasetDownloaderError> DatasetDownloader::downloadFromKaggle(
        const Str& datasetName, const Str& outputPath, const Str& apiToken) {
        SetRetT(void, DatasetDownloaderError);

        // Use HttpRequest to prepare the request
        HttpRequest request(Str("https://www.kaggle.com/api/v1/datasets/download/") + datasetName, HttpMethod::GET);
        request.addHeader(Str("Authorization"), Str("Bearer ") + apiToken);

        // Create an HttpClient and send the request
        std::unique_ptr<HttpResponse> response = HttpClient::send(request);

        if (!response) {
            Err(DatasetDownloaderError(Str("Failed to get HTTP response for Kaggle dataset: ") + datasetName));
        }

        // Check HTTP status code
        if (response->getStatusCode() >= 400) {
            Err(DatasetDownloaderError(Str("HTTP error: ") + Str(std::to_string(response->getStatusCode()))
                                       + Str(" for Kaggle dataset: ") + datasetName));
        }

        // Open output file
        std::ofstream outFile(outputPath.data(), std::ios::binary);
        if (!outFile.is_open()) {
            Err(DatasetDownloaderError(Str("Failed to open output file: ") + outputPath));
        }

        // Write the response body to the file
        outFile << response->getBody().c_str();
        outFile.close();

        Ok();
    }

} // namespace hahaha::ml
