//  Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//       https://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Contributors:
//  Napbad (napbad.sen@gmail.com) (https://github.com/Napbad)
//

//
// Created by napbad on 3/27/26.
//

#include <gtest/gtest.h>
#include <cstring>
#include <memory>

#include "backend/cpu/CPUMemoryManager.h"
#include "backend/CommonPointer.h"
#include "backend/Device.h"

namespace h3::core::backend::cpu {

/**
 * @brief Test fixture for CPUMemoryManager tests
 */
class CPUMemoryManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_memoryManager = std::make_unique<CPUMemoryManager>();
    }

    void TearDown() override {
        m_memoryManager.reset();
    }

    std::unique_ptr<CPUMemoryManager> m_memoryManager;
    static constexpr SizeT m_defaultSize = 1024;  // 1KB
};

/**
 * @brief Test allocation of various sizes
 */
TEST_F(CPUMemoryManagerTest, AllocateVariousSizes) {
    // Test small allocation
    auto smallResult = m_memoryManager->allocate(64);
    EXPECT_TRUE(smallResult.has_value());
    EXPECT_NE(smallResult->get(), nullptr);
    EXPECT_EQ(smallResult->size(), 64);
    EXPECT_EQ(smallResult->device().type(), DeviceType::CPU);
    m_memoryManager->deallocate(*smallResult);

    // Test default size allocation
    auto result = m_memoryManager->allocate(m_defaultSize);
    EXPECT_TRUE(result.has_value());
    EXPECT_NE(result->get(), nullptr);
    EXPECT_EQ(result->size(), m_defaultSize);
    EXPECT_EQ(result->device().type(), DeviceType::CPU);

    // Test large allocation
    auto largeResult = m_memoryManager->allocate(1024 * 1024);  // 1MB
    EXPECT_TRUE(largeResult.has_value());
    EXPECT_NE(largeResult->get(), nullptr);
    EXPECT_EQ(largeResult->size(), 1024 * 1024);
    
    // Cleanup
    m_memoryManager->deallocate(*result);
    m_memoryManager->deallocate(*largeResult);
}

/**
 * @brief Test deallocation
 */
TEST_F(CPUMemoryManagerTest, Deallocation) {
    auto ptr = m_memoryManager->allocate(m_defaultSize);
    ASSERT_TRUE(ptr.has_value());

    void* rawPtr = ptr->get();
    (void)rawPtr;
    EXPECT_NO_THROW(m_memoryManager->deallocate(*ptr));
}

/**
 * @brief Test copy between two buffers
 */
TEST_F(CPUMemoryManagerTest, CopyOperation) {
    const SizeT size = 256;
    auto src = m_memoryManager->allocate(size);
    auto dst = m_memoryManager->allocate(size);
    
    ASSERT_TRUE(src.has_value());
    ASSERT_TRUE(dst.has_value());
    
    // Fill source with pattern
    std::memset(src->as<char>(), 0xAB, size);
    
    // Perform copy
    auto copyResult = m_memoryManager->copy(*dst, *src, size);
    EXPECT_TRUE(copyResult.has_value());
    
    // Verify data was copied correctly
    EXPECT_EQ(std::memcmp(dst->as<char>(), src->as<char>(), size), 0);
    
    // Cleanup
    m_memoryManager->deallocate(*src);
    m_memoryManager->deallocate(*dst);
}

/**
 * @brief Test move operation with overlapping regions
 */
TEST_F(CPUMemoryManagerTest, MoveOverlapping) {
    const SizeT size = 512;
    auto buffer = m_memoryManager->allocate(size);
    ASSERT_TRUE(buffer.has_value());
    
    // Fill buffer with pattern
    char* bufferPtr = buffer->as<char>();
    for (SizeT i = 0; i < size; ++i) {
        bufferPtr[i] = static_cast<char>(i & 0xFF);
    }
    
    // Create overlapping pointers (offset by 128 bytes)
    CommonPointer src{bufferPtr + 128, Device{0, DeviceType::CPU}, size - 128};
    CommonPointer dst{bufferPtr, Device{0, DeviceType::CPU}, size - 128};
    
    // Perform move (should handle overlap correctly)
    auto moveResult = m_memoryManager->move(dst, src, size - 128);
    EXPECT_TRUE(moveResult.has_value());
    
    // Verify data was moved correctly
    for (SizeT i = 0; i < size - 128; ++i) {
        EXPECT_EQ(bufferPtr[i], static_cast<char>((i + 128) & 0xFF));
    }
    
    // Cleanup
    m_memoryManager->deallocate(*buffer);
}

/**
 * @brief Test copyFromHostToDevice (same as copy for CPU)
 */
TEST_F(CPUMemoryManagerTest, CopyFromHostToDevice) {
    const SizeT size = 128;
    auto host = m_memoryManager->allocate(size);
    auto device = m_memoryManager->allocate(size);
    
    ASSERT_TRUE(host.has_value());
    ASSERT_TRUE(device.has_value());
    
    // Fill host memory with pattern
    std::memset(host->as<char>(), 0xCD, size);
    
    // Copy from host to device
    auto result = m_memoryManager->copyFromHostToDevice(*device, *host, size);
    EXPECT_TRUE(result.has_value());
    
    // Verify data
    EXPECT_EQ(std::memcmp(device->as<char>(), host->as<char>(), size), 0);
    
    // Cleanup
    m_memoryManager->deallocate(*host);
    m_memoryManager->deallocate(*device);
}

/**
 * @brief Test copyFromDeviceToHost (same as copy for CPU)
 */
TEST_F(CPUMemoryManagerTest, CopyFromDeviceToHost) {
    const SizeT size = 128;
    auto host = m_memoryManager->allocate(size);
    auto device = m_memoryManager->allocate(size);
    
    ASSERT_TRUE(host.has_value());
    ASSERT_TRUE(device.has_value());
    
    // Fill device memory with pattern
    std::memset(device->as<char>(), 0xEF, size);
    
    // Copy from device to host
    auto result = m_memoryManager->copyFromDeviceToHost(*host, *device, size);
    EXPECT_TRUE(result.has_value());
    
    // Verify data
    EXPECT_EQ(std::memcmp(host->as<char>(), device->as<char>(), size), 0);
    
    // Cleanup
    m_memoryManager->deallocate(*host);
    m_memoryManager->deallocate(*device);
}

/**
 * @brief Test error handling with insufficient buffer size
 */
TEST_F(CPUMemoryManagerTest, CopyBufferTooSmall) {
    const SizeT srcSize = 64;
    const SizeT dstSize = 32;
    
    auto src = m_memoryManager->allocate(srcSize);
    auto dst = m_memoryManager->allocate(dstSize);
    
    ASSERT_TRUE(src.has_value());
    ASSERT_TRUE(dst.has_value());
    
    // Try to copy more than destination can hold
    auto result = m_memoryManager->copy(*dst, *src, srcSize);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), ErrorCode::MemoryError);
    
    // Cleanup
    m_memoryManager->deallocate(*src);
    m_memoryManager->deallocate(*dst);
}

/**
 * @brief Test error handling with insufficient buffer size for move
 */
TEST_F(CPUMemoryManagerTest, MoveBufferTooSmall) {
    const SizeT srcSize = 64;
    const SizeT dstSize = 32;
    
    auto src = m_memoryManager->allocate(srcSize);
    auto dst = m_memoryManager->allocate(dstSize);
    
    ASSERT_TRUE(src.has_value());
    ASSERT_TRUE(dst.has_value());
    
    // Try to move more than destination can hold
    auto result = m_memoryManager->move(*dst, *src, srcSize);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), ErrorCode::MemoryError);
    
    // Cleanup
    m_memoryManager->deallocate(*src);
    m_memoryManager->deallocate(*dst);
}

/**
 * @brief Test zero-size operations
 */
TEST_F(CPUMemoryManagerTest, ZeroSizeOperations) {
    auto ptr = m_memoryManager->allocate(0);
    ASSERT_TRUE(ptr.has_value());
    EXPECT_EQ(ptr->size(), 0);
    
    // Zero-size copy should succeed
    auto copyResult = m_memoryManager->copy(*ptr, *ptr, 0);
    EXPECT_TRUE(copyResult.has_value());
    
    // Zero-size move should succeed
    auto moveResult = m_memoryManager->move(*ptr, *ptr, 0);
    EXPECT_TRUE(moveResult.has_value());
    
    m_memoryManager->deallocate(*ptr);
}

/**
 * @brief Test CommonPointer helper methods
 */
TEST_F(CPUMemoryManagerTest, CommonPointerHelpers) {
    auto ptr = m_memoryManager->allocate(m_defaultSize);
    ASSERT_TRUE(ptr.has_value());
    
    // Test get()
    EXPECT_NE(ptr->get(), nullptr);
    
    // Test as<T>()
    int* intPtr = ptr->as<int>();
    EXPECT_NE(intPtr, nullptr);
    
    // Test device()
    EXPECT_EQ(ptr->device().type(), DeviceType::CPU);
    
    // Test size()
    EXPECT_EQ(ptr->size(), m_defaultSize);
    
    // Test to_string()
    std::string str = ptr->to_string();
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("CommonPointer"), std::string::npos);
    
    m_memoryManager->deallocate(*ptr);
}

/**
 * @brief Test multiple allocations and deallocations
 */
TEST_F(CPUMemoryManagerTest, MultipleAllocations) {
    const int numAllocs = 10;
    std::vector<std::expected<CommonPointer, Error>> ptrs;
    
    // Allocate multiple buffers
    for (int i = 0; i < numAllocs; ++i) {
        auto ptr = m_memoryManager->allocate(128);
        ASSERT_TRUE(ptr.has_value());
        ptrs.push_back(std::move(ptr));
    }
    
    // Fill each with different pattern
    for (int i = 0; i < numAllocs; ++i) {
        std::memset(ptrs[i]->as<char>(), static_cast<int>(i), 128);
    }
    
    // Verify patterns
    for (int i = 0; i < numAllocs; ++i) {
        char* data = ptrs[i]->as<char>();
        for (int j = 0; j < 128; ++j) {
            EXPECT_EQ(data[j], static_cast<char>(i));
        }
    }
    
    // Deallocate all
    for (int i = 0; i < numAllocs; ++i) {
        m_memoryManager->deallocate(*ptrs[i]);
    }
}

}  // namespace h3::core::backend::cpu
