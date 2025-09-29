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
#include <curl/curl.h>
#include <fstream>
#include <ml/util/dataset/DatasetDownloader.h>
#include <sstream>

namespace hahaha::ml {

  namespace {

    size_t writeCallback(void *contents, size_t size, size_t nmemb, void *userp) {
      auto *file = static_cast<std::ofstream *>(userp);
      size_t realSize = size * nmemb;
      file->write(static_cast<char *>(contents), realSize);
      return realSize;
    }

    int progressCallback(void *clientp, double dltotal, double dlnow, double ultotal, double ulnow) {
      (void) clientp;
      (void) dltotal;
      (void) dlnow;
      (void) ultotal;
      (void) ulnow;
      return 0; // Return non-zero to abort transfer
    }

  } // namespace

  Res<void, DatasetDownloaderError> DatasetDownloader::downloadFromUrl(const ds::Str &url, const ds::Str &outputPath,
                                                                       bool showProgress) {
    SetRetT(void, DatasetDownloaderError);

    // Initialize CURL
    CURL *curl = curl_easy_init();
    if (!curl) {
      Err(DatasetDownloaderError(ds::Str("Failed to initialize CURL")));
    }

    // Open output file
    std::ofstream outFile(outputPath.data(), std::ios::binary);
    if (!outFile) {
      curl_easy_cleanup(curl);
      Err(DatasetDownloaderError(ds::Str("Failed to open output file: ") + outputPath));
    }

    // Set CURL options
    curl_easy_setopt(curl, CURLOPT_URL, url.data());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &outFile);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);

    if (showProgress) {
      curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
      curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progressCallback);
    }

    // Perform the request
    CURLcode res = curl_easy_perform(curl);

    // Cleanup
    outFile.close();
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
      Err(DatasetDownloaderError(ds::Str("CURL error: ") + ds::Str(curl_easy_strerror(res))));
    }

    Ok();
  }

  Res<void, DatasetDownloaderError> DatasetDownloader::downloadFromUCI(const ds::Str &datasetName,
                                                                       const ds::Str &outputPath) {
    ds::Str baseUrl = ds::Str("https://archive.ics.uci.edu/ml/machine-learning-databases/");
    ds::Str url = baseUrl + datasetName + ds::Str("/") + datasetName + ds::Str(".data");
    return downloadFromUrl(url, outputPath);
  }

  Res<void, DatasetDownloaderError> DatasetDownloader::downloadFromKaggle(const ds::Str &datasetName,
                                                                          const ds::Str &outputPath,
                                                                          const ds::Str &apiToken) {
    SetRetT(void, DatasetDownloaderError);

    // Initialize CURL
    CURL *curl = curl_easy_init();
    if (!curl) {
      Err(DatasetDownloaderError(ds::Str("Failed to initialize CURL")));
    }

    // Open output file
    std::ofstream outFile(outputPath.data(), std::ios::binary);
    if (!outFile) {
      curl_easy_cleanup(curl);
      Err(DatasetDownloaderError(ds::Str("Failed to open output file: ") + outputPath));
    }

    // Set up headers
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, ("Authorization: Bearer " + apiToken).data());

    // Set CURL options
    ds::Str url = ds::Str("https://www.kaggle.com/api/v1/datasets/download/") + datasetName;
    curl_easy_setopt(curl, CURLOPT_URL, url.data());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &outFile);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);

    // Perform the request
    CURLcode res = curl_easy_perform(curl);

    // Cleanup
    outFile.close();
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
      Err(DatasetDownloaderError(ds::Str("CURL error: ") + ds::Str(curl_easy_strerror(res))));
    }

    Ok();
  }

} // namespace hahaha::ml
