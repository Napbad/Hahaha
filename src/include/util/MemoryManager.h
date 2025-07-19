#ifndef HIAHIAHIA_MEMORY_MANAGER_H
#define HIAHIAHIA_MEMORY_MANAGER_H

#include <memory>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <cstddef>

namespace hiahiahia {
namespace util {

/**
 * @brief Memory allocation strategy
 */
enum class AllocationStrategy {
    Direct,     ///< Direct allocation from system
    Pooled,     ///< Allocation from memory pool
    Cached,     ///< Cached allocation for reuse
    Pinned      ///< Pinned memory for faster transfer
};

/**
 * @brief Memory block information
 */
struct MemoryBlock {
    void* ptr;          ///< Memory pointer
    size_t size;        ///< Block size in bytes
    bool used;          ///< Usage status
    AllocationStrategy strategy; ///< Allocation strategy used
    
    MemoryBlock(void* p, size_t s, AllocationStrategy strat = AllocationStrategy::Direct)
        : ptr(p), size(s), used(false), strategy(strat) {}
};

/**
 * @brief Memory pool configuration
 */
struct MemoryPoolConfig {
    size_t initial_size;        ///< Initial pool size in bytes
    size_t max_size;            ///< Maximum pool size in bytes
    size_t block_size;          ///< Default block size in bytes
    size_t alignment;           ///< Memory alignment in bytes
    bool enable_growth;         ///< Allow pool growth
    
    MemoryPoolConfig(size_t init = 1024 * 1024, size_t max = 1024 * 1024 * 1024,
                     size_t block = 4096, size_t align = 64, bool growth = true)
        : initial_size(init), max_size(max), block_size(block), 
          alignment(align), enable_growth(growth) {}
};

/**
 * @brief Abstract memory manager interface
 */
class MemoryManager {
public:
    virtual ~MemoryManager() = default;
    
    /**
     * @brief Allocate memory with specified strategy
     * @param size Size in bytes to allocate
     * @param strategy Allocation strategy to use
     * @return Pointer to allocated memory, nullptr if failed
     */
    [[nodiscard]] virtual void* allocate(size_t size, AllocationStrategy strategy = AllocationStrategy::Direct) = 0;
    
    /**
     * @brief Deallocate memory
     * @param ptr Pointer to memory to deallocate
     */
    virtual void deallocate(void* ptr) = 0;
    
    /**
     * @brief Get total allocated memory
     * @return Total allocated memory in bytes
     */
    [[nodiscard]] virtual size_t get_total_allocated() const = 0;
    
    /**
     * @brief Get peak memory usage
     * @return Peak memory usage in bytes
     */
    [[nodiscard]] virtual size_t get_peak_usage() const = 0;
    
    /**
     * @brief Reset memory manager state
     */
    virtual void reset() = 0;
    
    /**
     * @brief Get memory statistics
     * @return Statistics as string
     */
    [[nodiscard]] virtual std::string get_statistics() const = 0;
};

/**
 * @brief CPU memory manager implementation
 */
class CPUMemoryManager : public MemoryManager {
public:
    explicit CPUMemoryManager(const MemoryPoolConfig& config = MemoryPoolConfig());
    ~CPUMemoryManager() override;
    
    // MemoryManager interface
    [[nodiscard]] void* allocate(size_t size, AllocationStrategy strategy = AllocationStrategy::Direct) override;
    void deallocate(void* ptr) override;
    [[nodiscard]] size_t get_total_allocated() const override;
    [[nodiscard]] size_t get_peak_usage() const override;
    void reset() override;
    [[nodiscard]] std::string get_statistics() const override;
    
    // CPU-specific methods
    [[nodiscard]] void* allocate_aligned(size_t size, size_t alignment);
    void deallocate_aligned(void* ptr);
    [[nodiscard]] size_t get_cache_line_size() const;
    
private:
    struct AllocationInfo {
        size_t size;
        AllocationStrategy strategy;
        bool is_aligned;
        size_t alignment;
        
        AllocationInfo(size_t s, AllocationStrategy strat, bool aligned = false, size_t align = 0)
            : size(s), strategy(strat), is_aligned(aligned), alignment(align) {}
    };
    
    MemoryPoolConfig config_;
    std::unordered_map<void*, AllocationInfo> allocations_;
    std::vector<MemoryBlock> memory_pool_;
    mutable std::mutex mutex_;
    size_t total_allocated_;
    size_t peak_usage_;
    
    void* allocate_from_pool(size_t size);
    void return_to_pool(void* ptr);
    void initialize_pool();
    void grow_pool(size_t additional_size);
};

/**
 * @brief GPU memory manager implementation
 */
class GPUMemoryManager : public MemoryManager {
public:
    explicit GPUMemoryManager(int device_id = 0, const MemoryPoolConfig& config = MemoryPoolConfig());
    ~GPUMemoryManager() override;
    
    // MemoryManager interface
    [[nodiscard]] void* allocate(size_t size, AllocationStrategy strategy = AllocationStrategy::Direct) override;
    void deallocate(void* ptr) override;
    [[nodiscard]] size_t get_total_allocated() const override;
    [[nodiscard]] size_t get_peak_usage() const override;
    void reset() override;
    [[nodiscard]] std::string get_statistics() const override;
    
    // GPU-specific methods
    [[nodiscard]] void* allocate_pinned(size_t size);
    void deallocate_pinned(void* ptr);
    [[nodiscard]] size_t get_device_memory() const;
    [[nodiscard]] size_t get_free_device_memory() const;
    
private:
    int device_id_;
    MemoryPoolConfig config_;
    std::unordered_map<void*, size_t> allocations_;
    std::unordered_map<void*, size_t> pinned_allocations_;
    mutable std::mutex mutex_;
    size_t total_allocated_;
    size_t peak_usage_;
    
#ifdef HIAHIAHIA_CUDA_ENABLED
    void initialize_cuda_device();
#endif
};

/**
 * @brief Global memory manager factory
 */
class MemoryManagerFactory {
public:
    /**
     * @brief Create memory manager for specific device type
     * @param device_type Device type
     * @param device_id Device identifier
     * @param config Memory pool configuration
     * @return Shared pointer to memory manager
     */
    [[nodiscard]] static std::shared_ptr<MemoryManager> create_memory_manager(
        int device_id = 0, 
        const MemoryPoolConfig& config = MemoryPoolConfig());
    
    /**
     * @brief Get singleton instance of CPU memory manager
     * @return Reference to CPU memory manager
     */
    static CPUMemoryManager& get_cpu_manager();
    
    /**
     * @brief Get singleton instance of GPU memory manager
     * @param device_id GPU device ID
     * @return Reference to GPU memory manager
     */
    static GPUMemoryManager& get_gpu_manager(int device_id = 0);
    
private:
    MemoryManagerFactory() = delete;
};

} // namespace util
} // namespace hiahiahia

#endif // HIAHIAHIA_MEMORY_MANAGER_H 