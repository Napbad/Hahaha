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

#include "dataset/DatasetDownloader.h"

#include <curl/curl.h>
#include <fstream>
#include <sstream>

#include "Error.h"
#include "Res.h"
#include "ds/String.h"
#include "io/file/fileUtil.h"

namespace hahaha::ml
{

using namespace hahaha::common;

size_t write_data(const void* ptr, const size_t size, const size_t nmemb, FILE* stream)
{
    const size_t written = fwrite(ptr, size, nmemb, stream);
    return written;
}

Res<void, DatasetDownloaderError> DatasetDownloader::downloadFromUrl(const String& url,
                                                                           const String& outputPath,
                                                                           const bool redownload)
{
    SetRetT(void, DatasetDownloaderError);

    if (util::io::fileExists(outputPath) && !redownload)
    {
        Ok();
    }

    FILE* fp = fopen(outputPath.cStr(), "wb");
    if (fp == nullptr)
    {
        Err(DatasetDownloaderError("Failed to open file for writing"));
    }

    if (CURL* curl = curl_easy_init())
    {
        curl_easy_setopt(curl, CURLOPT_URL, url.cStr());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);

        if (const CURLcode res = curl_easy_perform(curl); res != CURLE_OK)
        {
            fclose(fp);
            curl_easy_cleanup(curl);
            Err(DatasetDownloaderError(curl_easy_strerror(res)));
        }
        curl_easy_cleanup(curl);
    }
    fclose(fp);

    if (std::ifstream checkFile(outputPath.cStr()); checkFile.is_open())
    {
        char buffer[10] = {0};
        checkFile.read(buffer, 9);
        if (String(buffer).startsWith(String("NOT FOUND")))
        {
            checkFile.close();
            std::remove(outputPath.cStr());
            Err(DatasetDownloaderError("File not found on server (content starts with 'NOT FOUND')"));
        }
        checkFile.close();
    }

    Ok();
}

Res<void, DatasetDownloaderError> DatasetDownloader::downloadFromUCI(const String& dataset_name,
                                                                           const String& save_path,
                                                                           const bool redownload)
{

    const String base_url("https://archive.ics.uci.edu/ml/machine-learning-databases/");
    const String url = base_url + dataset_name + "/" + dataset_name + ".data";
    return downloadFromUrl(url, save_path, redownload);
}

Res<void, DatasetDownloaderError> DatasetDownloader::downloadFromKaggle(const String& dataset_name,
                                                                              const String& save_path,
                                                                              const String& api_token,
                                                                              const bool redownload)
{
    SetRetT(void, DatasetDownloaderError);
    Err(DatasetDownloaderError("Not implemented"));
}

} // namespace hahaha::ml
