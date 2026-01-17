#pragma once

/**
 * @file GraphicsCompat.h
 * @brief Compatibility layer for transitioning from OpenGL to NVRHI
 * 
 * This header provides helper functions to ease the transition from
 * direct OpenGL calls to the NVRHI graphics abstraction layer.
 */

#include "Graphics/GraphicsInit.h"
#include <string>
#include <memory>

namespace Graphics {

/**
 * @class GLToNVRHIAdapter
 * @brief Adapts OpenGL patterns to NVRHI equivalents
 */
class GLToNVRHIAdapter {
public:
    /**
     * Create a texture from image data (OpenGL glTexImage2D replacement)
     * 
     * @param width Texture width
     * @param height Texture height
     * @param format Pixel format
     * @param data Pixel data pointer
     * @return Created texture shared_ptr
     */
    static TexturePtr CreateTexture2D(uint32_t width, uint32_t height, 
                                     TextureFormat format, const void* data) {
        auto device = GetDevice();
        if (!device) return nullptr;

        TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.format = format;
        desc.initialData = const_cast<void*>(data);

        return device->CreateTexture(desc);
    }

    /**
     * Create a renderable texture (OpenGL glFramebufferTexture2D replacement)
     * 
     * @param width Texture width
     * @param height Texture height
     * @param format Pixel format
     * @param isColorTarget Is this a color render target?
     * @return Created texture shared_ptr
     */
    static TexturePtr CreateRenderTarget(uint32_t width, uint32_t height,
                                        TextureFormat format, bool isColorTarget = true) {
        auto device = GetDevice();
        if (!device) return nullptr;

        TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.format = format;
        desc.renderTarget = isColorTarget;

        return device->CreateTexture(desc);
    }

    /**
     * Create a depth texture (OpenGL depth attachment replacement)
     * 
     * @param width Texture width
     * @param height Texture height
     * @return Created depth texture shared_ptr
     */
    static TexturePtr CreateDepthTarget(uint32_t width, uint32_t height) {
        auto device = GetDevice();
        if (!device) return nullptr;

        TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.format = TextureFormat::D32_FLOAT;
        desc.depthTarget = true;

        return device->CreateTexture(desc);
    }

    /**
     * Create a vertex buffer (OpenGL glGenBuffers + glBufferData replacement)
     * 
     * @param size Buffer size in bytes
     * @param data Initial data pointer
     * @param isDynamic Can be updated frequently?
     * @return Created buffer shared_ptr
     */
    static BufferPtr CreateVertexBuffer(size_t size, const void* data, bool isDynamic = false) {
        auto device = GetDevice();
        if (!device) return nullptr;

        BufferDesc desc;
        desc.size = size;
        desc.usage = BufferUsage::VertexBuffer;
        desc.initialData = const_cast<void*>(data);
        desc.dynamic = isDynamic;
        desc.cpuAccess = isDynamic;

        return device->CreateBuffer(desc);
    }

    /**
     * Create an index buffer (OpenGL element buffer replacement)
     * 
     * @param size Buffer size in bytes
     * @param data Initial data pointer
     * @param isDynamic Can be updated frequently?
     * @return Created buffer shared_ptr
     */
    static BufferPtr CreateIndexBuffer(size_t size, const void* data, bool isDynamic = false) {
        auto device = GetDevice();
        if (!device) return nullptr;

        BufferDesc desc;
        desc.size = size;
        desc.usage = BufferUsage::IndexBuffer;
        desc.initialData = const_cast<void*>(data);
        desc.dynamic = isDynamic;
        desc.cpuAccess = isDynamic;

        return device->CreateBuffer(desc);
    }

    /**
     * Create a constant buffer (OpenGL uniform buffer replacement)
     * 
     * @param size Buffer size in bytes (usually aligned to 256 bytes)
     * @param data Initial data pointer
     * @return Created buffer shared_ptr
     */
    static BufferPtr CreateConstantBuffer(size_t size, const void* data = nullptr) {
        auto device = GetDevice();
        if (!device) return nullptr;

        // Align to D3D12 constant buffer requirement (256 bytes)
        size = (size + 255) & ~255;

        BufferDesc desc;
        desc.size = size;
        desc.usage = BufferUsage::ConstantBuffer;
        desc.initialData = const_cast<void*>(data);
        desc.cpuAccess = true;
        desc.dynamic = true;

        return device->CreateBuffer(desc);
    }

    /**
     * Create a structured buffer (OpenGL SSBO replacement)
     * 
     * @param elementSize Size of each element
     * @param elementCount Number of elements
     * @param data Initial data pointer
     * @return Created buffer shared_ptr
     */
    static BufferPtr CreateStructuredBuffer(size_t elementSize, uint32_t elementCount,
                                          const void* data = nullptr) {
        auto device = GetDevice();
        if (!device) return nullptr;

        BufferDesc desc;
        desc.size = elementSize * elementCount;
        desc.usage = BufferUsage::StructuredBuffer;
        desc.initialData = const_cast<void*>(data);

        return device->CreateBuffer(desc);
    }

    /**
     * Update buffer data (OpenGL glBufferSubData replacement)
     * 
     * @param buffer Buffer to update
     * @param data Data to write
     * @param size Data size in bytes
     * @param offset Offset in buffer
     */
    static void UpdateBuffer(BufferPtr buffer, const void* data, size_t size, 
                           size_t offset = 0) {
        if (buffer) {
            buffer->UpdateData(data, size, offset);
        }
    }

    /**
     * Map buffer for CPU access (OpenGL glMapBuffer replacement)
     * 
     * @param buffer Buffer to map
     * @return Mapped pointer, or nullptr on error
     */
    static void* MapBuffer(BufferPtr buffer) {
        if (buffer) {
            return buffer->Map();
        }
        return nullptr;
    }

    /**
     * Unmap buffer after CPU access (OpenGL glUnmapBuffer replacement)
     * 
     * @param buffer Buffer to unmap
     */
    static void UnmapBuffer(BufferPtr buffer) {
        if (buffer) {
            buffer->Unmap();
        }
    }

    /**
     * Load shader from bytecode (OpenGL glCompileShader replacement)
     * 
     * @param bytecode Compiled shader bytecode
     * @param type Shader stage type
     * @param language Shader language (HLSL/GLSL/SPIR-V)
     * @return Created shader shared_ptr
     */
    static ShaderPtr CreateShaderFromBytecode(const std::vector<uint8_t>& bytecode,
                                             ShaderType type,
                                             ShaderLanguage language = ShaderLanguage::HLSL) {
        auto device = GetDevice();
        if (!device) return nullptr;

        ShaderDesc desc;
        desc.bytecode = bytecode;
        desc.type = type;
        desc.language = language;
        desc.entryPoint = "main";

        return device->CreateShader(desc);
    }

    /**
     * Get device screen dimensions
     * 
     * @param width Output width
     * @param height Output height
     */
    static void GetScreenDimensions(uint32_t& width, uint32_t& height) {
        auto device = GetDevice();
        if (device) {
            width = device->GetScreenWidth();
            height = device->GetScreenHeight();
        } else {
            width = height = 0;
        }
    }
};

/**
 * @brief Create a complete NVRHI setup from window handle
 * 
 * Useful for initializing NVRHI in Application or Window class
 * 
 * @param windowHandle Platform window handle (HWND on Windows)
 * @param width Window width
 * @param height Window height
 * @param backend Graphics backend to use
 * @return true on success, false on failure
 */
inline bool InitializeNVRHIFromWindow(void* windowHandle, uint32_t width, 
                                      uint32_t height, GraphicsBackend backend) {
    auto device = InitializeGraphics(backend, width, height, windowHandle);
    return device != nullptr;
}

} // namespace Graphics

// Compatibility macros for quick migration
#define CREATE_TEXTURE_2D(w, h, fmt, data) \
    Graphics::GLToNVRHIAdapter::CreateTexture2D(w, h, fmt, data)

#define CREATE_RENDER_TARGET(w, h, fmt) \
    Graphics::GLToNVRHIAdapter::CreateRenderTarget(w, h, fmt, true)

#define CREATE_DEPTH_TARGET(w, h) \
    Graphics::GLToNVRHIAdapter::CreateDepthTarget(w, h)

#define CREATE_VERTEX_BUFFER(size, data) \
    Graphics::GLToNVRHIAdapter::CreateVertexBuffer(size, data, false)

#define CREATE_INDEX_BUFFER(size, data) \
    Graphics::GLToNVRHIAdapter::CreateIndexBuffer(size, data, false)

#define CREATE_CONSTANT_BUFFER(size, data) \
    Graphics::GLToNVRHIAdapter::CreateConstantBuffer(size, data)

#define GET_GRAPHICS_DEVICE() Graphics::GetDevice()

#define GET_DEVICE_SCREEN_SIZE(w, h) Graphics::GLToNVRHIAdapter::GetScreenDimensions(w, h)
