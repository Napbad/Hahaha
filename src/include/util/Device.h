#ifndef HIAHIAHIA_DEVICE_H
#define HIAHIAHIA_DEVICE_H

#include <string>
#include <string_view>
#include <memory>
#include <vector>
#include <optional>

namespace hiahiahia {
namespace util {

/**
 * @brief Device types supported by the framework
 */
enum class DeviceType {
    CPU,    ///< Central Processing Unit
    GPU,    ///< Graphics Processing Unit (CUDA/OpenCL)
    FPGA,   ///< Field Programmable Gate Array
    NPU     ///< Neural Processing Unit
};

/**
 * @brief Computational capabilities of devices
 */
enum class DeviceCapability {
    Float32,        ///< 32-bit floating point support
    Float64,        ///< 64-bit floating point support
    Int32,          ///< 32-bit integer support
    Int64,          ///< 64-bit integer support
    HalfPrecision,  ///< 16-bit floating point support
    MixedPrecision, ///< Mixed precision computation
    ParallelCompute,///< Parallel computation support
    MemoryPool,     ///< Memory pooling capability
    SIMD,           ///< Single Instruction Multiple Data
    TensorCores     ///< Tensor core acceleration
};

/**
 * @brief Device information structure
 */
struct DeviceInfo {
    std::string name;                           ///< Device name
    DeviceType type;                            ///< Device type
    size_t memory_size;                         ///< Total memory in bytes
    size_t compute_units;                       ///< Number of compute units
    std::vector<DeviceCapability> capabilities; ///< Supported capabilities
    bool available;                             ///< Device availability
    int device_id;                              ///< Device identifier
    
    DeviceInfo() = default;
    DeviceInfo(std::string n, DeviceType t, size_t mem, size_t units, 
               std::vector<DeviceCapability> caps, bool avail, int id = 0)
        : name(std::move(n)), type(t), memory_size(mem), compute_units(units), 
          capabilities(std::move(caps)), available(avail), device_id(id) {}
};

/**
 * @brief Abstract base class for all compute devices
 * 
 * This class defines the interface that all device implementations must follow.
 * It provides unified access to device capabilities, memory management, and
 * data transfer operations.
 */
class Device {
public:
    virtual ~Device() = default;
    
    // ==================== Basic Information ====================
    
    /**
     * @brief Get device name
     * @return Device name as string view
     */
    [[nodiscard]] virtual std::string_view name() const = 0;
    
    /**
     * @brief Get device type
     * @return Device type enumeration
     */
    [[nodiscard]] virtual DeviceType type() const = 0;
    
    /**
     * @brief Get detailed device information
     * @return Device information structure
     */
    [[nodiscard]] virtual DeviceInfo info() const = 0;
    
    /**
     * @brief Check if device is available for use
     * @return True if device is available
     */
    [[nodiscard]] virtual bool is_available() const = 0;
    
    /**
     * @brief Get device identifier
     * @return Device ID
     */
    [[nodiscard]] virtual int get_device_id() const = 0;
    
    // ==================== Memory Management ====================
    
    /**
     * @brief Allocate memory on device
     * @param size Size in bytes to allocate
     * @return Pointer to allocated memory, nullptr if failed
     */
    [[nodiscard]] virtual void* allocate(size_t size) = 0;
    
    /**
     * @brief Deallocate memory on device
     * @param ptr Pointer to memory to deallocate
     */
    virtual void deallocate(void* ptr) = 0;
    
    /**
     * @brief Get current memory usage
     * @return Memory usage in bytes
     */
    [[nodiscard]] virtual size_t get_memory_usage() const = 0;
    
    /**
     * @brief Get free memory available
     * @return Free memory in bytes
     */
    [[nodiscard]] virtual size_t get_free_memory() const = 0;
    
    /**
     * @brief Allocate memory from pool (if supported)
     * @param size Size in bytes to allocate
     * @return Pointer to allocated memory, nullptr if failed
     */
    [[nodiscard]] virtual void* allocate_pooled(size_t size) = 0;
    
    /**
     * @brief Deallocate memory to pool
     * @param ptr Pointer to memory to deallocate
     */
    virtual void deallocate_pooled(void* ptr) = 0;
    
    // ==================== Data Transfer ====================
    
    /**
     * @brief Copy data from host to device
     * @param device_ptr Destination device pointer
     * @param host_ptr Source host pointer
     * @param size Size in bytes to copy
     */
    virtual void copy_host_to_device(void* device_ptr, const void* host_ptr, size_t size) = 0;
    
    /**
     * @brief Copy data from device to host
     * @param host_ptr Destination host pointer
     * @param device_ptr Source device pointer
     * @param size Size in bytes to copy
     */
    virtual void copy_device_to_host(void* host_ptr, const void* device_ptr, size_t size) = 0;
    
    /**
     * @brief Copy data between devices
     * @param dst Destination device pointer
     * @param src Source device pointer
     * @param size Size in bytes to copy
     */
    virtual void copy_device_to_device(void* dst, const void* src, size_t size) = 0;
    
    // ==================== Synchronization ====================
    
    /**
     * @brief Synchronize device operations
     */
    virtual void synchronize() = 0;
    
    /**
     * @brief Create execution stream
     * @return Stream handle, nullptr if failed
     */
    [[nodiscard]] virtual void* create_stream() = 0;
    
    /**
     * @brief Destroy execution stream
     * @param stream Stream handle to destroy
     */
    virtual void destroy_stream(void* stream) = 0;
    
    /**
     * @brief Synchronize specific stream
     * @param stream Stream handle to synchronize
     */
    virtual void synchronize_stream(void* stream) = 0;
    
    // ==================== Device Management ====================
    
    /**
     * @brief Set this device as current
     */
    virtual void set_device() = 0;
    
    /**
     * @brief Reset device state
     */
    virtual void reset_device() = 0;
    
    // ==================== Capability Queries ====================
    
    /**
     * @brief Check if device supports specific capability
     * @param cap Capability to check
     * @return True if capability is supported
     */
    [[nodiscard]] virtual bool supports_capability(DeviceCapability cap) const = 0;
    
    /**
     * @brief Check if device supports specific data type
     * @param dtype Data type name (e.g., "float32", "int64")
     * @return True if data type is supported
     */
    [[nodiscard]] virtual bool supports_data_type(const std::string& dtype) const = 0;
    
    /**
     * @brief Get optimal memory alignment for this device
     * @return Memory alignment in bytes
     */
    [[nodiscard]] virtual size_t get_optimal_alignment() const = 0;
};

// 设备管理器
class DeviceManager {
public:
    static DeviceManager& get_instance();
    
    // 设备发现和管理
    [[nodiscard]] std::vector<std::shared_ptr<Device>> discover_devices();
    [[nodiscard]] std::shared_ptr<Device> get_device(DeviceType type, int id = 0);
    [[nodiscard]] std::shared_ptr<Device> get_best_device(DeviceType type);
    [[nodiscard]] std::vector<std::shared_ptr<Device>> get_devices_by_type(DeviceType type);
    
    // 设备选择
    void set_default_device(std::shared_ptr<Device> device);
    [[nodiscard]] std::shared_ptr<Device> get_default_device() const;
    
    // 设备信息
    [[nodiscard]] std::vector<DeviceInfo> get_all_device_info() const;
    void print_device_info() const;
    
    // 内存管理
    [[nodiscard]] size_t get_total_memory() const;
    [[nodiscard]] size_t get_used_memory() const;
    void clear_memory_pools();
    
private:
    DeviceManager() = default;
    ~DeviceManager() = default;
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;
    
    std::vector<std::shared_ptr<Device>> devices_;
    std::shared_ptr<Device> default_device_;
    bool initialized_ = false;
    
    void initialize_devices();
};

// 设备工厂
class DeviceFactory {
public:
    [[nodiscard]] static std::shared_ptr<Device> create_device(DeviceType type, int id = 0);
    [[nodiscard]] static std::shared_ptr<Device> create_device(const std::string& device_name);
    
private:
    DeviceFactory() = delete;
};

} // namespace util
} // namespace hiahiahia

#endif // HIAHIAHIA_DEVICE_H 