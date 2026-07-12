// Simple test for Scalar::convertToType
#include <iostream>
#include <cassert>
#include <cmath>

#include "math/Scalar.h"
#include "backend/Device.h"

using namespace h3::core;
using namespace h3::core::math;

int main() {
    backend::Device cpu = backend::Device(0, backend::DeviceType::CPU);
    
    std::cout << "Test 1: Int32 to Float32..." << std::endl;
    {
        auto s = Scalar::zeros(DataType::Int32, cpu);
        *s.as<Int32>() = 42;
        s.convertToType(DataType::Float32);
        assert(s.dtype() == DataType::Float32);
        assert(std::abs(*s.as<Float32>() - 42.0f) < 0.001f);
        std::cout << "  PASSED" << std::endl;
    }
    
    std::cout << "Test 2: Float32 to Int32..." << std::endl;
    {
        auto s = Scalar::zeros(DataType::Float32, cpu);
        *s.as<Float32>() = 3.7f;
        s.convertToType(DataType::Int32);
        assert(s.dtype() == DataType::Int32);
        assert(*s.as<Int32>() == 4);
        std::cout << "  PASSED" << std::endl;
    }
    
    std::cout << "Test 3: Float64 to Float32..." << std::endl;
    {
        auto s = Scalar::zeros(DataType::Float64, cpu);
        *s.as<Float64>() = 3.14159265358979;
        s.convertToType(DataType::Float32);
        assert(s.dtype() == DataType::Float32);
        assert(std::abs(*s.as<Float32>() - 3.14159f) < 0.0001f);
        std::cout << "  PASSED" << std::endl;
    }
    
    std::cout << "Test 4: Int32 to Int64..." << std::endl;
    {
        auto s = Scalar::zeros(DataType::Int32, cpu);
        *s.as<Int32>() = 12345;
        s.convertToType(DataType::Int64);
        assert(s.dtype() == DataType::Int64);
        assert(*s.as<Int64>() == 12345);
        std::cout << "  PASSED" << std::endl;
    }
    
    std::cout << "Test 5: Same type (no-op)..." << std::endl;
    {
        auto s = Scalar::zeros(DataType::Float32, cpu);
        *s.as<Float32>() = 1.5f;
        s.convertToType(DataType::Float32);
        assert(s.dtype() == DataType::Float32);
        assert(std::abs(*s.as<Float32>() - 1.5f) < 0.001f);
        std::cout << "  PASSED" << std::endl;
    }
    
    std::cout << "Test 6: UInt8 to Int32..." << std::endl;
    {
        auto s = Scalar::zeros(DataType::UInt8, cpu);
        *s.as<UInt8>() = 200;
        s.convertToType(DataType::Int32);
        assert(s.dtype() == DataType::Int32);
        assert(*s.as<Int32>() == 200);
        std::cout << "  PASSED" << std::endl;
    }
    
    std::cout << "\nAll tests PASSED!" << std::endl;
    return 0;
}
