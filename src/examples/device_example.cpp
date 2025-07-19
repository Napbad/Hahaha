#include "util/Device.h"
#include "util/DeviceManager.h"
#include <iostream>
#include <format>

int main() {
    std::cout << "=== Device Management Example ===" << std::endl;
    
    try {
        // 获取设备管理器实例
        auto& device_manager = hiahiahia::util::DeviceManager::get_instance();
        
        // 打印所有设备信息
        device_manager.print_device_info();
        
        // 获取默认设备
        auto default_device = device_manager.get_default_device();
        if (default_device) {
            std::cout << std::format("Default device: {} (ID: {})", 
                                    default_device->name(), 
                                    default_device->get_device_id()) << std::endl;
        }
        
        // 获取特定类型的设备
        auto cpu_devices = device_manager.get_devices_by_type(hiahiahia::util::DeviceType::CPU);
        auto gpu_devices = device_manager.get_devices_by_type(hiahiahia::util::DeviceType::GPU);
        
        std::cout << std::format("Found {} CPU devices and {} GPU devices", 
                                cpu_devices.size(), gpu_devices.size()) << std::endl;
        
        // 获取最佳GPU设备
        auto best_gpu = device_manager.get_best_device(hiahiahia::util::DeviceType::GPU);
        if (best_gpu) {
            auto info = best_gpu->info();
            std::cout << std::format("Best GPU: {} with {} MB memory", 
                                    info.name, info.memory_size / (1024 * 1024)) << std::endl;
        }
        
        // 使用设备进行内存分配和操作
        if (default_device) {
            std::cout << "\n=== Memory Operations ===" << std::endl;
            
            // 分配内存
            size_t size = 1024 * sizeof(float);
            void* ptr = default_device->allocate(size);
            if (ptr) {
                std::cout << std::format("Allocated {} bytes on {}", size, default_device->name()) << std::endl;
                
                // 获取内存使用情况
                std::cout << std::format("Memory usage: {} bytes", default_device->get_memory_usage()) << std::endl;
                std::cout << std::format("Free memory: {} bytes", default_device->get_free_memory()) << std::endl;
                
                // 释放内存
                default_device->deallocate(ptr);
                std::cout << "Memory deallocated" << std::endl;
            }
        }
        
        // 使用设备工厂创建设备
        std::cout << "\n=== Device Factory ===" << std::endl;
        
        auto cpu_device = hiahiahia::util::DeviceFactory::create_device(hiahiahia::util::DeviceType::CPU, 0);
        if (cpu_device) {
            std::cout << std::format("Created CPU device: {}", cpu_device->name()) << std::endl;
        }
        
        auto gpu_device = hiahiahia::util::DeviceFactory::create_device(hiahiahia::util::DeviceType::GPU, 0);
        if (gpu_device) {
            std::cout << std::format("Created GPU device: {}", gpu_device->name()) << std::endl;
        }
        
        // 通过名称创建设备
        auto device_by_name = hiahiahia::util::DeviceFactory::create_device("GPU0");
        if (device_by_name) {
            std::cout << std::format("Created device by name: {}", device_by_name->name()) << std::endl;
        }
        
        // 检查设备能力
        if (default_device) {
            std::cout << "\n=== Device Capabilities ===" << std::endl;
            
            auto capabilities = {
                hiahiahia::util::DeviceCapability::Float32,
                hiahiahia::util::DeviceCapability::Float64,
                hiahiahia::util::DeviceCapability::Int32,
                hiahiahia::util::DeviceCapability::ParallelCompute
            };
            
            for (auto cap : capabilities) {
                bool supported = default_device->supports_capability(cap);
                std::cout << std::format("Capability {}: {}", static_cast<int>(cap), 
                                        supported ? "Supported" : "Not Supported") << std::endl;
            }
            
            // 检查数据类型支持
            auto data_types = {"float32", "float64", "int32", "int64"};
            for (const auto& dtype : data_types) {
                bool supported = default_device->supports_data_type(dtype);
                std::cout << std::format("Data type {}: {}", dtype, 
                                        supported ? "Supported" : "Not Supported") << std::endl;
            }
        }
        
        std::cout << "\n=== Example completed successfully ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 