#pragma once

#include <memory>
#include <vector>
#include <string>
#include <glm/glm.hpp>

namespace Graphics {

// Forward declarations
class Device;
class CommandList;
class Buffer;
class Texture;
class Shader;
class Pipeline;

// Common enums and types
enum class GraphicsBackend {
    D3D12,
    D3D11,
    Vulkan
};

enum class BufferUsage {
    ConstantBuffer,
    VertexBuffer,
    IndexBuffer,
    StructuredBuffer,
    RWStructuredBuffer,
    IndirectBuffer
};

enum class TextureFormat {
    R8G8B8A8_UNORM,
    R16G16B16A16_FLOAT,
    R32G32B32A32_FLOAT,
    R32_FLOAT,
    R32G32_FLOAT,
    D32_FLOAT,
    D24_UNORM_S8_UINT
};

enum class ResourceAccess {
    Read,
    Write,
    ReadWrite,
    ColorAttachment,
    DepthAttachment
};

enum class ShaderType {
    Vertex,
    Fragment,
    Compute,
    Geometry,
    TessControl,
    TessEval
};

enum class ShaderLanguage {
    HLSL,
    GLSL,
    SPIR_V
};

struct BufferDesc {
    size_t size = 0;
    BufferUsage usage = BufferUsage::VertexBuffer;
    bool cpuAccess = false;
    bool dynamic = false;
    void* initialData = nullptr;
};

struct TextureDesc {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t mipLevels = 1;
    uint32_t arraySize = 1;
    TextureFormat format = TextureFormat::R8G8B8A8_UNORM;
    bool renderTarget = false;
    bool depthTarget = false;
    bool staging = false;
    void* initialData = nullptr;
};

struct ShaderDesc {
    ShaderType type = ShaderType::Vertex;
    ShaderLanguage language = ShaderLanguage::HLSL;
    std::string entryPoint = "main";
    std::vector<uint8_t> bytecode;
    std::string source;
};

struct ViewportRect {
    float x = 0.0f;
    float y = 0.0f;
    float width = 800.0f;
    float height = 600.0f;
    float minDepth = 0.0f;
    float maxDepth = 1.0f;
};

struct ClearColor {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 1.0f;
};

// Smart pointers
using DevicePtr = std::shared_ptr<Device>;
using CommandListPtr = std::shared_ptr<CommandList>;
using BufferPtr = std::shared_ptr<Buffer>;
using TexturePtr = std::shared_ptr<Texture>;
using ShaderPtr = std::shared_ptr<Shader>;
using PipelinePtr = std::shared_ptr<Pipeline>;

} // namespace Graphics
