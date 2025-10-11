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

// TODO: This test is disabled because it fails in CI due to network issues.
// TEST_F(DatasetDownloaderTest, DownloadFromUrlTest)
// {
//     const ds::String url = "https://example.com";
//     const ds::String dest = "example.html";
// 
//     auto result = DatasetDownloader::downloadFromUrl(url, dest);
//     ASSERT_TRUE(result.isOk()) << "Download should succeed";
// 
//     // Check if file exists
//     std::ifstream file(dest.c_str());
//     EXPECT_TRUE(file.good()) << "Destination file should be created";
//     file.close();
// 
//     // Clean up
//     std::remove(dest.c_str());
// }

// TEST_F(DatasetDownloaderTest, DownloadFromUrlFailTest)
// {
//     const ds::String url("invalid-url");
//     const ds::String dest("test_download.txt");
// 
//     EXPECT_THROW(DatasetDownloader::downloadFromUrl(url, dest, false), std::runtime_error);
// 
//     // Verify file was not created or is empty
//     if (std::ifstream file(dest.c_str()); file.is_open())
//     {
//         file.seekg(0, std::ios::end);
//         EXPECT_EQ(file.tellg(), 0) << "File should be empty on download failure";
//         file.close();
//         std::remove(dest.c_str());
//     }
// }

TEST_F(DatasetDownloaderTest, DownloadFromUCITest)
{
    EXPECT_THROW(DatasetDownloader::downloadFromUCI(ds::String("non-existent-dataset"), ds::String("test_download.txt")), std::runtime_error);
}

TEST_F(DatasetDownloaderTest, DownloadFromKaggleTest)
{
    EXPECT_THROW(
        DatasetDownloader::downloadFromKaggle(
            ds::String("some-dataset"),
            ds::String("test_download.txt"),
            ds::String("fake-api-token")),
        std::runtime_error);
}

TEST_F(DatasetDownloaderTest, DownloadToInvalidPathTest)
{
    EXPECT_THROW(
        DatasetDownloader::downloadFromUrl(
            ds::String("http://example.com"),
            ds::String("/non_existent_dir/test.txt")),
        std::runtime_error);
}
