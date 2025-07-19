#ifndef HIAHIAHIA_DEVICE_MANAGER_H
#define HIAHIAHIA_DEVICE_MANAGER_H

#include "Device.h"
#include "CPUDevice.h"
#include "GPUDevice.h"
#include <iostream>
#include <algorithm>
#include <format>

namespace hiahiahia {
namespace util {

// 设备管理器实现
inline DeviceManager& DeviceManager::get_instance() {
    static DeviceManager instance;
    if (!instance.initialized_) {
        instance.initialize_devices();
    }
    return instance;
}

inline std::vector<std::shared_ptr<Device>> DeviceManager::discover_devices() {
    if (!initialized_) {
        initialize_devices();
    }
    return devices_;
}

inline std::shared_ptr<Device> DeviceManager::get_device(DeviceType type, int id) {
    if (!initialized_) {
        initialize_devices();
    }
    
    for (const auto& device : devices_) {
        if (device->type() == type && device->get_device_id() == id) {
            return device;
        }
    }
    return nullptr;
}

inline std::shared_ptr<Device> DeviceManager::get_best_device(DeviceType type) {
    if (!initialized_) {
        initialize_devices();
    }
    
    std::shared_ptr<Device> best_device = nullptr;
    size_t best_memory = 0;
    
    for (const auto& device : devices_) {
        if (device->type() == type && device->is_available()) {
            auto info = device->info();
            if (info.memory_size > best_memory) {
                best_memory = info.memory_size;
                best_device = device;
            }
        }
    }
    
    return best_device;
}

inline std::vector<std::shared_ptr<Device>> DeviceManager::get_devices_by_type(DeviceType type) {
    if (!initialized_) {
        initialize_devices();
    }
    
    std::vector<std::shared_ptr<Device>> result;
    for (const auto& device : devices_) {
        if (device->type() == type) {
            result.push_back(device);
        }
    }
    return result;
}

inline void DeviceManager::set_default_device(std::shared_ptr<Device> device) {
    default_device_ = std::move(device);
}

inline std::shared_ptr<Device> DeviceManager::get_default_device() const {
    return default_device_;
}

inline std::vector<DeviceInfo> DeviceManager::get_all_device_info() const {
    std::vector<DeviceInfo> info;
    for (const auto& device : devices_) {
        info.push_back(device->info());
    }
    return info;
}

inline void DeviceManager::print_device_info() const {
    std::cout << "=== Device Information ===" << std::endl;
    
    for (const auto& device : devices_) {
        auto info = device->info();
        std::cout << std::format("Device: {} (ID: {})", info.name, device->get_device_id()) << std::endl;
        std::cout << std::format("  Type: {}", static_cast<int>(info.type)) << std::endl;
        std::cout << std::format("  Memory: {} MB", info.memory_size / (1024 * 1024)) << std::endl;
        std::cout << std::format("  Compute Units: {}", info.compute_units) << std::endl;
        std::cout << std::format("  Available: {}", info.available ? "Yes" : "No") << std::endl;
        
        std::cout << "  Capabilities: ";
        for (size_t i = 0; i < info.capabilities.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << static_cast<int>(info.capabilities[i]);
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }
    
    if (default_device_) {
        std::cout << std::format("Default Device: {} (ID: {})", 
                                default_device_->name(), 
                                default_device_->get_device_id()) << std::endl;
    }
}

inline size_t DeviceManager::get_total_memory() const {
    size_t total = 0;
    for (const auto& device : devices_) {
        total += device->info().memory_size;
    }
    return total;
}

inline size_t DeviceManager::get_used_memory() const {
    size_t used = 0;
    for (const auto& device : devices_) {
        used += device->get_memory_usage();
    }
    return used;
}

inline void DeviceManager::clear_memory_pools() {
    for (auto& device : devices_) {
        device->reset_device();
    }
}

// 设备工厂实现
inline std::shared_ptr<Device> DeviceFactory::create_device(DeviceType type, int id) {
    switch (type) {
        case DeviceType::CPU:
            return std::make_shared<CPUDevice>(id);
        case DeviceType::GPU:
            return std::make_shared<GPUDevice>(id);
        case DeviceType::FPGA:
        case DeviceType::NPU:
            // TODO: 实现FPGA和NPU设备
            return nullptr;
        default:
            return nullptr;
    }
}

inline std::shared_ptr<Device> DeviceFactory::create_device(const std::string& device_name) {
    if (device_name == "CPU" || device_name == "cpu") {
        return std::make_shared<CPUDevice>(0);
    } else if (device_name.substr(0, 3) == "GPU" || device_name.substr(0, 3) == "gpu") {
        int id = 0;
        if (device_name.length() > 3) {
            try {
                id = std::stoi(device_name.substr(3));
            } catch (...) {
                id = 0;
            }
        }
        return std::make_shared<GPUDevice>(id);
    }
    return nullptr;
}

// 设备管理器初始化实现
inline void DeviceManager::initialize_devices() {
    if (initialized_) return;
    
    // 添加CPU设备
    devices_.push_back(std::make_shared<CPUDevice>(0));
    
    // 尝试添加GPU设备
    int gpu_count = 0;
#ifdef HIAHIAHIA_CUDA_ENABLED
    cudaError_t err = cudaGetDeviceCount(&gpu_count);
    if (err == cudaSuccess) {
        for (int i = 0; i < gpu_count; ++i) {
            auto gpu_device = std::make_shared<GPUDevice>(i);
            if (gpu_device->is_available()) {
                devices_.push_back(gpu_device);
            }
        }
    }
#endif
    
    // 设置默认设备（优先GPU，否则CPU）
    if (!devices_.empty()) {
        auto gpu_device = get_best_device(DeviceType::GPU);
        if (gpu_device) {
            default_device_ = gpu_device;
        } else {
            default_device_ = devices_[0]; // CPU
        }
    }
    
    initialized_ = true;
}

} // namespace util
} // namespace hiahiahia

#endif // HIAHIAHIA_DEVICE_MANAGER_H 