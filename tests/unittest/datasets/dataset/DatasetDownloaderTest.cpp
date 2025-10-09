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

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

#include <ds/String.h>
#include <dataset/DatasetDownloader.h>

using namespace hahaha::ml;
using namespace hahaha::common;

class DatasetDownloaderTest : public ::testing::Test
{
  public:
    void SetUp() override
    {
        // Make sure test file doesn't exist
        std::remove("test_download.txt");
    }

    void TearDown() override
    {
        // Clean up test file
        std::remove("test_download.txt");
    }
};

TEST_F(DatasetDownloaderTest, DownloadFromUrlTest)
{
    // This test would require an actual internet connection and a valid URL
    // For unit testing purposes, we'll test with a known invalid URL to verify error handling

    auto res = DatasetDownloader::downloadFromUrl(
        String("http://invalid-url-that-does-not-exist-12345.com/test.txt"), String("test_download.txt"), false);

    // Should fail with an error
    EXPECT_TRUE(res.isErr());

    // Verify file was not created or is empty
    if (std::ifstream file("test_download.txt"); file.is_open())
    {
        file.seekg(0, std::ios::end);
        EXPECT_EQ(file.tellg(), 0); // File should be empty
        file.close();
    }
}

TEST_F(DatasetDownloaderTest, DownloadFromUCITest)
{
    // Test with a non-existent UCI dataset to verify error handling
    const auto res = DatasetDownloader::downloadFromUCI(ds::String("non-existent-dataset"), ds::String("test_download.txt"));

    // Should fail with an error
    EXPECT_TRUE(res.isErr());
}

TEST_F(DatasetDownloaderTest, DownloadFromKaggleTest)
{
    // Test with a non-existent Kaggle dataset to verify error handling
    const auto res = DatasetDownloader::downloadFromKaggle(
        ds::String("non-existent-dataset/non-existent-file"), ds::String("test_download.txt"), ds::String("fake-api-token"));

    // Should fail with an error
    EXPECT_TRUE(res.isErr());
}

TEST_F(DatasetDownloaderTest, DownloadToInvalidPathTest)
{
    // Try to download to an invalid path
    const auto res = DatasetDownloader::downloadFromUrl(
        ds::String("http://httpbin.org/get"), ds::String("/invalid/path/that/should/not/exist/test.txt"), false);

    // Should fail with an error
    EXPECT_TRUE(res.isErr());
}
