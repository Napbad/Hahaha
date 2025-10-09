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

#ifndef HIAHIAHIA_DATASETDOWNLOADER_H
#define HIAHIAHIA_DATASETDOWNLOADER_H

// Standard Library
#include <utility>

// Project
#include "Error.h"
#include "Res.h"
#include "ds/String.h"

namespace hahaha::ml
{

using namespace hahaha::common;
/**
 * Dataset downloader error class
 */
class DatasetDownloaderError final : public Error
{
  public:
    explicit DatasetDownloaderError(ds::String message, ds::String location = ds::String("DatasetDownloader"))
        : message_(std::move(message)), location_(std::move(location))
    {
    }

    explicit DatasetDownloaderError(const char* message, ds::String location = ds::String("DatasetDownloader"))
        : message_(ds::String(message)), location_(std::move(location))
    {
    }

    [[nodiscard]] ds::String typeName() const override
    {
        return ds::String("DatasetDownloaderError");
    }
    [[nodiscard]] ds::String message() const override
    {
        return message_;
    }
    [[nodiscard]] ds::String location() const override
    {
        return location_;
    }
    [[nodiscard]] ds::String toString() const override
    {
        return typeName() + ds::String(": ") + message() + ds::String(" at ") + location();
    }

  private:
    ds::String message_;
    ds::String location_;
};

/**
 * Dataset downloader class
 *
 * Provides functionality to download datasets from various sources:
 * - Direct URL
 * - UCI Machine Learning Repository
 * - Kaggle
 */
class DatasetDownloader
{
  public:
    /**
     * Download a dataset from a direct URL
     */
    static Res<void, DatasetDownloaderError>
    downloadFromUrl(const ds::String& url, const ds::String& outputPath, bool redownload = false);

    /**
     * Download a dataset from UCI Machine Learning Repository
     */
    static Res<void, DatasetDownloaderError>
    downloadFromUCI(const ds::String& datasetName, const ds::String& outputPath, bool redownload = false);

    /**
     * Download a dataset from Kaggle
     */
    static Res<void, DatasetDownloaderError>
    downloadFromKaggle(const ds::String& datasetName, const ds::String& outputPath, const ds::String& apiToken, bool redownload = false);
};

} // namespace hahaha::ml

#endif // HIAHIAHIA_DATASETDOWNLOADER_H
