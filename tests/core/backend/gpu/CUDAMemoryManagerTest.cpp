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
#include <cuda_runtime.h>

#include "backend/cuda/CUDAMemoryManager.h"
#include "backend/CommonPointer.h"
#include "backend/Device.h"

namespace h3::core::backend::cuda {

/**
 * @brief Check if CUDA device is available
 */
bool isCudaAvailable() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0);
}

/**
 * @brief Test fixture for CUDAMemoryManager tests
 */
class CUDAMemoryManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!isCudaAvailable()) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
        m_memoryManager = std::make_unique<CUDAMemoryManager>();
    }

    void TearDown() override {
        m_memoryManager.reset();
        // Synchronize to ensure all CUDA operations are complete
        cudaDeviceSynchronize();
    }

    std::unique_ptr<CUDAMemoryManager> m_memoryManager;
    static constexpr SizeT m_defaultSize = 1024;  // 1KB
};

/**
 * @brief Test allocation of various sizes
 */
TEST_F(CUDAMemoryManagerTest, AllocateVariousSizes) {
    // Test small allocation
    auto smallResult = m_memoryManager->allocate(64);
    EXPECT_TRUE(smallResult.has_value());
    EXPECT_NE(smallResult->get(), nullptr);
    EXPECT_EQ(smallResult->size(), 64);
    EXPECT_EQ(smallResult->device().type(), DeviceType::CUDA);
    m_memoryManager->deallocate(*smallResult);

    // Test default size allocation
    auto result = m_memoryManager->allocate(m_defaultSize);
    EXPECT_TRUE(result.has_value());
    EXPECT_NE(result->get(), nullptr);
    EXPECT_EQ(result->size(), m_defaultSize);
    EXPECT_EQ(result->device().type(), DeviceType::CUDA);

    // Test large allocation (1MB)
    auto largeResult = m_memoryManager->allocate(1024 * 1024);
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
TEST_F(CUDAMemoryManagerTest, Deallocate) {
    auto ptr = m_memoryManager->allocate(m_defaultSize);
    ASSERT_TRUE(ptr.has_value());
    
    void* rawPtr = ptr->get();
    EXPECT_NO_THROW(m_memoryManager->deallocate(*ptr));
    
    // Test deallocation of nullptr (should be safe)
    CommonPointer nullPtr{nullptr, Device{0, DeviceType::CUDA}, 0};
    EXPECT_NO_THROW(m_memoryManager->deallocate(nullPtr));
}

/**
 * @brief Test copy operation (device-to-device)
 */
TEST_F(CUDAMemoryManagerTest, CopyDeviceToDevice) {
    const SizeT size = 256;
    auto src = m_memoryManager->allocate(size);
    auto dst = m_memoryManager->allocate(size);
    
    ASSERT_TRUE(src.has_value());
    ASSERT_TRUE(dst.has_value());
    
    // Fill source with pattern using host buffer
    std::vector<char> hostBuffer(size, 0xAB);
    auto h2dResult = m_memoryManager->copyFromHostToDevice(*src, 
        CommonPointer{hostBuffer.data(), Device{0, DeviceType::CPU}, size}, 
        size);
    ASSERT_TRUE(h2dResult.has_value());
    
    // Perform device-to-device copy
    auto copyResult = m_memoryManager->copy(*dst, *src, size);
    EXPECT_TRUE(copyResult.has_value());
    
    // Verify data was copied correctly by copying back to host
    std::vector<char> verifyBuffer(size, 0);
    auto d2hResult = m_memoryManager->copyFromDeviceToHost(
        CommonPointer{verifyBuffer.data(), Device{0, DeviceType::CPU}, size},
        *dst, size);
    ASSERT_TRUE(d2hResult.has_value());
    
    EXPECT_EQ(std::memcmp(verifyBuffer.data(), hostBuffer.data(), size), 0);
    
    // Cleanup
    m_memoryManager->deallocate(*src);
    m_memoryManager->deallocate(*dst);
}

/**
 * @brief Test move operation (device-to-device)
 */
TEST_F(CUDAMemoryManagerTest, MoveDeviceToDevice) {
    const SizeT size = 512;
    auto src = m_memoryManager->allocate(size);
    auto dst = m_memoryManager->allocate(size);
    
    ASSERT_TRUE(src.has_value());
    ASSERT_TRUE(dst.has_value());
    
    // Fill source with pattern
    std::vector<char> hostData(size);
    for (SizeT i = 0; i < size; ++i) {
        hostData[i] = static_cast<char>(i & 0xFF);
    }
    
    auto h2dResult = m_memoryManager->copyFromHostToDevice(*src,
        CommonPointer{hostData.data(), Device{0, DeviceType::CPU}, size},
        size);
    ASSERT_TRUE(h2dResult.has_value());
    
    // Perform move
    auto moveResult = m_memoryManager->move(*dst, *src, size);
    EXPECT_TRUE(moveResult.has_value());
    
    // Verify data was moved
    std::vector<char> verifyBuffer(size, 0);
    auto d2hResult = m_memoryManager->copyFromDeviceToHost(
        CommonPointer{verifyBuffer.data(), Device{0, DeviceType::CPU}, size},
        *dst, size);
    ASSERT_TRUE(d2hResult.has_value());
    
    EXPECT_EQ(std::memcmp(verifyBuffer.data(), hostData.data(), size), 0);
    
    // Cleanup
    m_memoryManager->deallocate(*src);
    m_memoryManager->deallocate(*dst);
}

/**
 * @brief Test copyFromHostToDevice
 */
TEST_F(CUDAMemoryManagerTest, CopyFromHostToDevice) {
    const SizeT size = 128;
    auto device = m_memoryManager->allocate(size);
    ASSERT_TRUE(device.has_value());
    
    // Create host buffer with pattern
    std::vector<char> hostBuffer(size, 0xCD);
    CommonPointer hostPtr{hostBuffer.data(), Device{0, DeviceType::CPU}, size};
    
    // Copy from host to device
    auto result = m_memoryManager->copyFromHostToDevice(*device, hostPtr, size);
    EXPECT_TRUE(result.has_value());
    
    // Verify by copying back to host
    std::vector<char> verifyBuffer(size, 0);
    CommonPointer verifyPtr{verifyBuffer.data(), Device{0, DeviceType::CPU}, size};
    auto d2hResult = m_memoryManager->copyFromDeviceToHost(verifyPtr, *device, size);
    ASSERT_TRUE(d2hResult.has_value());
    
    EXPECT_EQ(std::memcmp(verifyBuffer.data(), hostBuffer.data(), size), 0);
    
    // Cleanup
    m_memoryManager->deallocate(*device);
}

/**
 * @brief Test copyFromDeviceToHost
 */
TEST_F(CUDAMemoryManagerTest, CopyFromDeviceToHost) {
    const SizeT size = 128;
    auto device = m_memoryManager->allocate(size);
    ASSERT_TRUE(device.has_value());
    
    // Fill device memory from host
    std::vector<char> hostBuffer(size, 0xEF);
    CommonPointer hostPtr{hostBuffer.data(), Device{0, DeviceType::CPU}, size};
    auto h2dResult = m_memoryManager->copyFromHostToDevice(*device, hostPtr, size);
    ASSERT_TRUE(h2dResult.has_value());
    
    // Copy from device to host
    std::vector<char> verifyBuffer(size, 0);
    CommonPointer verifyPtr{verifyBuffer.data(), Device{0, DeviceType::CPU}, size};
    auto result = m_memoryManager->copyFromDeviceToHost(verifyPtr, *device, size);
    EXPECT_TRUE(result.has_value());
    
    // Verify data
    EXPECT_EQ(std::memcmp(verifyBuffer.data(), hostBuffer.data(), size), 0);
    
    // Cleanup
    m_memoryManager->deallocate(*device);
}

/**
 * @brief Test error handling with insufficient buffer size
 */
TEST_F(CUDAMemoryManagerTest, CopyInsufficientBufferSize) {
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
 * @brief Test error handling for host-to-device with insufficient size
 */
TEST_F(CUDAMemoryManagerTest, HostToDeviceInsufficientSize) {
    const SizeT deviceSize = 32;
    const SizeT hostSize = 64;
    
    auto device = m_memoryManager->allocate(deviceSize);
    ASSERT_TRUE(device.has_value());
    
    std::vector<char> hostBuffer(hostSize, 0xAB);
    CommonPointer hostPtr{hostBuffer.data(), Device{0, DeviceType::CPU}, hostSize};
    
    // Try to copy more than device can hold
    auto result = m_memoryManager->copyFromHostToDevice(*device, hostPtr, hostSize);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), ErrorCode::MemoryError);
    
    // Cleanup
    m_memoryManager->deallocate(*device);
}

/**
 * @brief Test error handling for device-to-host with insufficient size
 */
TEST_F(CUDAMemoryManagerTest, DeviceToHostInsufficientSize) {
    const SizeT deviceSize = 32;
    const SizeT hostSize = 64;
    
    auto device = m_memoryManager->allocate(deviceSize);
    ASSERT_TRUE(device.has_value());
    
    std::vector<char> hostBuffer(hostSize, 0);
    CommonPointer hostPtr{hostBuffer.data(), Device{0, DeviceType::CPU}, hostSize};
    
    // Try to copy more than device has
    auto result = m_memoryManager->copyFromDeviceToHost(hostPtr, *device, hostSize);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), ErrorCode::MemoryError);
    
    // Cleanup
    m_memoryManager->deallocate(*device);
}

/**
 * @brief Test zero-size operations
 */
TEST_F(CUDAMemoryManagerTest, ZeroSizeOperations) {
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
 * @brief Test CommonPointer helper methods with CUDA device
 */
TEST_F(CUDAMemoryManagerTest, CommonPointerHelpers) {
    auto ptr = m_memoryManager->allocate(m_defaultSize);
    ASSERT_TRUE(ptr.has_value());
    
    // Test get()
    EXPECT_NE(ptr->get(), nullptr);
    
    // Test as<T>()
    float* floatPtr = ptr->as<float>();
    EXPECT_NE(floatPtr, nullptr);
    
    // Test device()
    EXPECT_EQ(ptr->device().type(), DeviceType::CUDA);
    
    // Test size()
    EXPECT_EQ(ptr->size(), m_defaultSize);
    
    // Test to_string()
    std::string str = ptr->to_string();
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("CommonPointer"), std::string::npos);
    EXPECT_NE(str.find("CUDA"), std::string::npos);
    
    m_memoryManager->deallocate(*ptr);
}

/**
 * @brief Test multiple allocations and deallocations
 */
TEST_F(CUDAMemoryManagerTest, MultipleAllocations) {
    const int numAllocs = 5;
    std::vector<std::expected<CommonPointer, Error>> ptrs;
    
    // Allocate multiple buffers
    for (int i = 0; i < numAllocs; ++i) {
        auto ptr = m_memoryManager->allocate(256);
        ASSERT_TRUE(ptr.has_value());
        ptrs.push_back(std::move(ptr));
    }
    
    // Fill each with different pattern using host buffers
    for (int i = 0; i < numAllocs; ++i) {
        std::vector<char> hostData(256, static_cast<char>(i));
        CommonPointer hostPtr{hostData.data(), Device{0, DeviceType::CPU}, 256};
        auto result = m_memoryManager->copyFromHostToDevice(*ptrs[i], hostPtr, 256);
        ASSERT_TRUE(result.has_value());
    }
    
    // Verify each allocation by copying back to host
    for (int i = 0; i < numAllocs; ++i) {
        std::vector<char> verifyBuffer(256, 0);
        CommonPointer hostPtr{verifyBuffer.data(), Device{0, DeviceType::CPU}, 256};
        auto result = m_memoryManager->copyFromDeviceToHost(hostPtr, *ptrs[i], 256);
        ASSERT_TRUE(result.has_value());
        
        for (int j = 0; j < 256; ++j) {
            EXPECT_EQ(verifyBuffer[j], static_cast<char>(i));
        }
    }
    
    // Deallocate all
    for (int i = 0; i < numAllocs; ++i) {
        m_memoryManager->deallocate(*ptrs[i]);
    }
}

/**
 * @brief Test round-trip host-to-device-to-host
 */
TEST_F(CUDAMemoryManagerTest, RoundTripTransfer) {
    const SizeT size = 512;
    auto device = m_memoryManager->allocate(size);
    ASSERT_TRUE(device.has_value());
    
    // Create host buffer with known pattern
    std::vector<char> originalData(size);
    for (SizeT i = 0; i < size; ++i) {
        originalData[i] = static_cast<char>((i * 7 + 13) & 0xFF);
    }
    
    // Copy to device
    CommonPointer hostSrc{originalData.data(), Device{0, DeviceType::CPU}, size};
    auto h2dResult = m_memoryManager->copyFromHostToDevice(*device, hostSrc, size);
    ASSERT_TRUE(h2dResult.has_value());
    
    // Copy back to host
    std::vector<char> receivedData(size, 0);
    CommonPointer hostDst{receivedData.data(), Device{0, DeviceType::CPU}, size};
    auto d2hResult = m_memoryManager->copyFromDeviceToHost(hostDst, *device, size);
    ASSERT_TRUE(d2hResult.has_value());
    
    // Verify data integrity
    EXPECT_EQ(std::memcmp(originalData.data(), receivedData.data(), size), 0);
    
    // Cleanup
    m_memoryManager->deallocate(*device);
}

/**
 * @brief Test chaining multiple operations
 */
TEST_F(CUDAMemoryManagerTest, ChainedOperations) {
    const SizeT size = 256;
    auto buf1 = m_memoryManager->allocate(size);
    auto buf2 = m_memoryManager->allocate(size);
    auto buf3 = m_memoryManager->allocate(size);
    
    ASSERT_TRUE(buf1.has_value());
    ASSERT_TRUE(buf2.has_value());
    ASSERT_TRUE(buf3.has_value());
    
    // Fill buf1 from host
    std::vector<char> testData(size, 0x42);
    CommonPointer hostPtr{testData.data(), Device{0, DeviceType::CPU}, size};
    auto h2dResult = m_memoryManager->copyFromHostToDevice(*buf1, hostPtr, size);
    ASSERT_TRUE(h2dResult.has_value());
    
    // Copy buf1 -> buf2
    auto copy1Result = m_memoryManager->copy(*buf2, *buf1, size);
    ASSERT_TRUE(copy1Result.has_value());
    
    // Move buf2 -> buf3
    auto moveResult = m_memoryManager->move(*buf3, *buf2, size);
    ASSERT_TRUE(moveResult.has_value());
    
    // Verify final data in buf3
    std::vector<char> verifyBuffer(size, 0);
    CommonPointer verifyPtr{verifyBuffer.data(), Device{0, DeviceType::CPU}, size};
    auto d2hResult = m_memoryManager->copyFromDeviceToHost(verifyPtr, *buf3, size);
    ASSERT_TRUE(d2hResult.has_value());
    
    EXPECT_EQ(std::memcmp(verifyBuffer.data(), testData.data(), size), 0);
    
    // Cleanup
    m_memoryManager->deallocate(*buf1);
    m_memoryManager->deallocate(*buf2);
    m_memoryManager->deallocate(*buf3);
}

}  // namespace h3::core::backend::cuda
