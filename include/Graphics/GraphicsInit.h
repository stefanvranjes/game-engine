#pragma once

#include "GraphicsDevice.h"
#include "NVRHIBackend.h"
#include <iostream>

namespace Graphics {

/**
 * @brief Initialize the graphics system with NVRHI
 * @param backend Graphics backend to use (D3D12, D3D11, or Vulkan)
 * @param width Screen width
 * @param height Screen height
 * @param windowHandle HWND or equivalent platform window handle
 * @return Initialized graphics device, or nullptr on failure
 */
inline DevicePtr InitializeGraphics(
    GraphicsBackend backend,
    uint32_t width,
    uint32_t height,
    void* windowHandle) {
    
    std::cout << "Initializing NVRHI Graphics System..." << std::endl;
    
    auto device = CreateDevice(backend);
    if (!device) {
        std::cerr << "Failed to create graphics device" << std::endl;
        return nullptr;
    }

    if (!device->Initialize(backend, width, height, windowHandle)) {
        std::cerr << "Failed to initialize graphics device" << std::endl;
        return nullptr;
    }

    std::cout << "Graphics system initialized successfully" << std::endl;
    std::cout << "  Backend: " << static_cast<int>(backend) << std::endl;
    std::cout << "  Resolution: " << width << "x" << height << std::endl;
    
    return device;
}

/**
 * @brief Shutdown the graphics system
 */
inline void ShutdownGraphics() {
    auto device = GetDevice();
    if (device) {
        device->Shutdown();
        SetDevice(nullptr);
    }
}

/**
 * @brief Get the currently active graphics device
 */
inline DevicePtr GetGraphicsDevice() {
    return GetDevice();
}

/**
 * @brief Get backend name for logging/display
 */
inline std::string GetBackendName(GraphicsBackend backend) {
    switch (backend) {
        case GraphicsBackend::D3D12:
            return "Direct3D 12";
        case GraphicsBackend::D3D11:
            return "Direct3D 11";
        case GraphicsBackend::Vulkan:
            return "Vulkan";
        default:
            return "Unknown";
    }
}

} // namespace Graphics
