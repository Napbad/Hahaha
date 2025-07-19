#ifndef HIAHIAHIA_GPU_DEVICE_H
#define HIAHIAHIA_GPU_DEVICE_H

#include "Device.h"
#include <unordered_map>
#include <mutex>
#include <vector>

#ifdef HIAHIAHIA_CUDA_ENABLED
#include <cuda_runtime.h>
#include <cuda.h>
#endif

namespace hiahiahia {
namespace util {

// GPU设备实现
class GPUDevice : public Device {
public:
    explicit GPUDevice(int device_id = 0);
    ~GPUDevice() override;
    
    // 基本信息
    [[nodiscard]] std::string_view name() const override;
    [[nodiscard]] DeviceType type() const override { return DeviceType::GPU; }
    [[nodiscard]] DeviceInfo info() const override;
    [[nodiscard]] bool is_available() const override;
    
    // 内存管理
    [[nodiscard]] void* allocate(size_t size) override;
    void deallocate(void* ptr) override;
    [[nodiscard]] size_t get_memory_usage() const override;
    [[nodiscard]] size_t get_free_memory() const override;
    
    // 数据传输
    void copy_host_to_device(void* device_ptr, const void* host_ptr, size_t size) override;
    void copy_device_to_host(void* host_ptr, const void* device_ptr, size_t size) override;
    void copy_device_to_device(void* dst, const void* src, size_t size) override;
    
    // 同步和异步
    void synchronize() override;
    void* create_stream() override;
    void destroy_stream(void* stream) override;
    void synchronize_stream(void* stream) override;
    
    // 设备管理
    void set_device() override;
    void reset_device() override;
    [[nodiscard]] int get_device_id() const override { return device_id_; }
    
    // 能力检查
    [[nodiscard]] bool supports_capability(DeviceCapability cap) const override;
    [[nodiscard]] bool supports_data_type(const std::string& dtype) const override;
    
    // GPU特有功能
    [[nodiscard]] int get_compute_capability_major() const;
    [[nodiscard]] int get_compute_capability_minor() const;
    [[nodiscard]] size_t get_max_threads_per_block() const;
    [[nodiscard]] size_t get_max_blocks_per_sm() const;
    [[nodiscard]] size_t get_multiprocessor_count() const;
    
private:
    int device_id_;
    size_t memory_usage_;
    std::unordered_map<void*, size_t> allocated_memory_;
    mutable std::mutex memory_mutex_;
    
#ifdef HIAHIAHIA_CUDA_ENABLED
    cudaDeviceProp device_properties_;
    std::vector<cudaStream_t> streams_;
    mutable std::mutex stream_mutex_;
    
    void initialize_cuda_device();
    [[nodiscard]] cudaStream_t get_or_create_stream();
#endif
    
    bool cuda_available_;
};

// GPU内存管理器
class GPUMemoryManager {
public:
    static GPUMemoryManager& get_instance();
    
    [[nodiscard]] void* allocate(size_t size, int device_id = 0);
    void deallocate(void* ptr, int device_id = 0);
    [[nodiscard]] size_t get_total_allocated(int device_id = 0) const;
    void reset(int device_id = 0);
    
    // 内存池管理
    [[nodiscard]] void* allocate_pooled(size_t size, int device_id = 0);
    void deallocate_pooled(void* ptr, int device_id = 0);
    
private:
    GPUMemoryManager() = default;
    ~GPUMemoryManager() = default;
    GPUMemoryManager(const GPUMemoryManager&) = delete;
    GPUMemoryManager& operator=(const GPUMemoryManager&) = delete;
    
    struct DeviceMemoryInfo {
        std::unordered_map<void*, size_t> allocations;
        size_t total_allocated = 0;
        mutable std::mutex mutex;
    };
    
    std::unordered_map<int, DeviceMemoryInfo> device_memory_;
    mutable std::mutex global_mutex_;
};

// GPU流管理器
class GPUStreamManager {
public:
    static GPUStreamManager& get_instance();
    
    [[nodiscard]] void* create_stream(int device_id = 0);
    void destroy_stream(void* stream, int device_id = 0);
    void synchronize_stream(void* stream, int device_id = 0);
    void synchronize_device(int device_id = 0);
    
private:
    GPUStreamManager() = default;
    ~GPUStreamManager() = default;
    GPUStreamManager(const GPUStreamManager&) = delete;
    GPUStreamManager& operator=(const GPUStreamManager&) = delete;
    
    struct DeviceStreamInfo {
        std::vector<void*> streams;
        mutable std::mutex mutex;
    };
    
    std::unordered_map<int, DeviceStreamInfo> device_streams_;
    mutable std::mutex global_mutex_;
};

// GPU事件管理器
class GPUEventManager {
public:
    static GPUEventManager& get_instance();
    
    [[nodiscard]] void* create_event(int device_id = 0);
    void destroy_event(void* event, int device_id = 0);
    void record_event(void* event, void* stream, int device_id = 0);
    void synchronize_event(void* event, int device_id = 0);
    [[nodiscard]] float get_event_elapsed_time(void* start_event, void* end_event, int device_id = 0);
    
private:
    GPUEventManager() = default;
    ~GPUEventManager() = default;
    GPUEventManager(const GPUEventManager&) = delete;
    GPUEventManager& operator=(const GPUEventManager&) = delete;
    
    struct DeviceEventInfo {
        std::vector<void*> events;
        mutable std::mutex mutex;
    };
    
    std::unordered_map<int, DeviceEventInfo> device_events_;
    mutable std::mutex global_mutex_;
};

} // namespace util
} // namespace hiahiahia

#endif // HIAHIAHIA_GPU_DEVICE_H 