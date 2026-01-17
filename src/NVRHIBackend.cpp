#include "Graphics/NVRHIBackend.h"
#include <iostream>
#include <cassert>
#include <glm/glm.hpp>

#ifdef _WIN32
#include <d3d12.h>
#include <dxgi1_4.h>
#endif

namespace Graphics {

// Global device instance
static DevicePtr g_Device = nullptr;

// ============================================================================
// NVRHIBuffer Implementation
// ============================================================================

NVRHIBuffer::NVRHIBuffer(nvrhi::IBuffer* nvrhi_buffer, const BufferDesc& desc)
    : m_NVRHIBuffer(nvrhi_buffer), m_Desc(desc) {
}

NVRHIBuffer::~NVRHIBuffer() {
    if (m_MappedPtr) {
        Unmap();
    }
}

void* NVRHIBuffer::Map() {
    if (!m_NVRHIBuffer) return nullptr;
    
    m_MappedPtr = m_NVRHIBuffer->map(nvrhi::CpuAccessMode::Read);
    return m_MappedPtr;
}

void NVRHIBuffer::Unmap() {
    if (!m_NVRHIBuffer || !m_MappedPtr) return;
    m_NVRHIBuffer->unmap();
    m_MappedPtr = nullptr;
}

void NVRHIBuffer::UpdateData(const void* data, size_t size, size_t offset) {
    if (!m_NVRHIBuffer || !data) return;
    
    // For dynamic buffers, map and update
    if (m_Desc.dynamic) {
        void* mappedPtr = Map();
        if (mappedPtr) {
            std::memcpy(static_cast<uint8_t*>(mappedPtr) + offset, data, size);
            Unmap();
        }
    }
}

// ============================================================================
// NVRHITexture Implementation
// ============================================================================

NVRHITexture::NVRHITexture(nvrhi::ITexture* nvrhi_texture, const TextureDesc& desc)
    : m_NVRHITexture(nvrhi_texture), m_Desc(desc) {
}

NVRHITexture::~NVRHITexture() {
}

void NVRHITexture::SetData(const void* data, size_t size, uint32_t mipLevel, uint32_t arrayIndex) {
    if (!m_NVRHITexture || !data) return;
    
    // Create a staging buffer for the texture data
    nvrhi::BufferDesc stagingDesc;
    stagingDesc.byteSize = size;
    stagingDesc.cpuAccess = nvrhi::CpuAccessMode::Write;
    stagingDesc.debugName = "TextureStagingBuffer";
    
    auto stagingBuffer = GetDevice()->CreateBuffer(
        BufferDesc{size, BufferUsage::StructuredBuffer, true, true, const_cast<void*>(data)}
    );
}

void* NVRHITexture::GetData(uint32_t mipLevel, uint32_t arrayIndex) {
    if (!m_NVRHITexture) return nullptr;
    
    // Implementation would involve reading back from GPU
    return nullptr;
}

void NVRHITexture::GenerateMipMaps() {
    if (!m_NVRHITexture) return;
    // This would be implemented using compute shaders or hardware mipmap generation
}

// ============================================================================
// NVRHIShader Implementation
// ============================================================================

NVRHIShader::NVRHIShader(nvrhi::IShader* nvrhi_shader, const ShaderDesc& desc)
    : m_NVRHIShader(nvrhi_shader), m_Desc(desc) {
}

NVRHIShader::~NVRHIShader() {
}

// ============================================================================
// NVRHIPipeline Implementation
// ============================================================================

NVRHIPipeline::NVRHIPipeline(nvrhi::GraphicsPipelineHandle pipeline)
    : m_Pipeline(pipeline) {
}

NVRHIPipeline::~NVRHIPipeline() {
}

void NVRHIPipeline::SetVertexShader(ShaderPtr shader) {
    m_VertexShader = shader;
}

void NVRHIPipeline::SetFragmentShader(ShaderPtr shader) {
    m_FragmentShader = shader;
}

void NVRHIPipeline::SetComputeShader(ShaderPtr shader) {
    m_ComputeShader = shader;
}

// ============================================================================
// NVRHICommandList Implementation
// ============================================================================

NVRHICommandList::NVRHICommandList(nvrhi::ICommandList* cmdList, nvrhi::IDevice* device)
    : m_CommandList(cmdList), m_Device(device) {
}

NVRHICommandList::~NVRHICommandList() {
}

void NVRHICommandList::Begin() {
    if (m_CommandList) {
        m_CommandList->open();
    }
}

void NVRHICommandList::End() {
    if (m_CommandList) {
        m_CommandList->close();
    }
}

void NVRHICommandList::Submit() {
    if (m_Device && m_CommandList) {
        m_Device->executeCommandList(m_CommandList);
    }
}

void NVRHICommandList::SetViewport(const ViewportRect& viewport) {
    if (!m_CommandList) return;
    
    nvrhi::Viewport vp;
    vp.minX = viewport.x;
    vp.minY = viewport.y;
    vp.maxX = viewport.x + viewport.width;
    vp.maxY = viewport.y + viewport.height;
    vp.minZ = viewport.minDepth;
    vp.maxZ = viewport.maxDepth;
    
    m_CommandList->setViewports(&vp, 1);
}

void NVRHICommandList::SetScissorRect(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
    if (!m_CommandList) return;
    
    nvrhi::Rect rect{static_cast<int>(x), static_cast<int>(y), 
                    static_cast<int>(x + width), static_cast<int>(y + height)};
    m_CommandList->setScissorRects(&rect, 1);
}

void NVRHICommandList::SetRenderTargets(const std::vector<TexturePtr>& renderTargets, TexturePtr depthTarget) {
    // Implementation for setting render targets
}

void NVRHICommandList::ClearRenderTarget(TexturePtr target, const ClearColor& color) {
    // Implementation for clearing render target
}

void NVRHICommandList::ClearDepthTarget(TexturePtr target, float depth) {
    // Implementation for clearing depth target
}

void NVRHICommandList::SetPipeline(PipelinePtr pipeline) {
    if (!m_CommandList || !pipeline) return;
    auto nvrhiPipeline = std::dynamic_pointer_cast<NVRHIPipeline>(pipeline);
    if (nvrhiPipeline) {
        m_CommandList->setPipeline(nvrhiPipeline->GetNVRHIPipeline());
    }
}

void NVRHICommandList::SetConstantBuffer(uint32_t slot, BufferPtr buffer) {
    // Implementation for setting constant buffer
}

void NVRHICommandList::SetVertexBuffer(uint32_t slot, BufferPtr buffer, uint32_t stride) {
    // Implementation for setting vertex buffer
}

void NVRHICommandList::SetIndexBuffer(BufferPtr buffer, bool is32Bit) {
    // Implementation for setting index buffer
}

void NVRHICommandList::SetTexture(uint32_t slot, TexturePtr texture) {
    // Implementation for setting texture
}

void NVRHICommandList::SetSampler(uint32_t slot, uint32_t samplerHandle) {
    // Implementation for setting sampler
}

void NVRHICommandList::DrawIndexed(uint32_t indexCount, uint32_t startIndexLocation, int32_t baseVertexLocation) {
    if (!m_CommandList) return;
    m_CommandList->drawIndexed(nvrhi::DrawArguments{indexCount, 1, startIndexLocation, baseVertexLocation, 0});
}

void NVRHICommandList::Draw(uint32_t vertexCount, uint32_t startVertexLocation) {
    if (!m_CommandList) return;
    m_CommandList->draw(nvrhi::DrawArguments{vertexCount, 1, 0, 0, startVertexLocation});
}

void NVRHICommandList::DrawInstanced(uint32_t vertexCountPerInstance, uint32_t instanceCount, uint32_t startVertexLocation, uint32_t startInstanceLocation) {
    if (!m_CommandList) return;
    m_CommandList->draw(nvrhi::DrawArguments{vertexCountPerInstance, instanceCount, 0, 0, startVertexLocation});
}

void NVRHICommandList::DrawIndexedInstanced(uint32_t indexCountPerInstance, uint32_t instanceCount, uint32_t startIndexLocation, int32_t baseVertexLocation, uint32_t startInstanceLocation) {
    if (!m_CommandList) return;
    m_CommandList->drawIndexed(nvrhi::DrawArguments{indexCountPerInstance, instanceCount, startIndexLocation, baseVertexLocation, startInstanceLocation});
}

void NVRHICommandList::Dispatch(uint32_t threadGroupCountX, uint32_t threadGroupCountY, uint32_t threadGroupCountZ) {
    if (!m_CommandList) return;
    m_CommandList->dispatch(threadGroupCountX, threadGroupCountY, threadGroupCountZ);
}

void NVRHICommandList::TransitionResource(TexturePtr resource, ResourceAccess oldAccess, ResourceAccess newAccess) {
    // Resource state transitions are often implicit in NVRHI
}

void NVRHICommandList::TransitionBuffer(BufferPtr resource, ResourceAccess oldAccess, ResourceAccess newAccess) {
    // Resource state transitions are often implicit in NVRHI
}

void NVRHICommandList::CopyBuffer(BufferPtr src, BufferPtr dst, size_t size, size_t srcOffset, size_t dstOffset) {
    if (!m_CommandList || !src || !dst) return;
    // Implementation for buffer copy
}

void NVRHICommandList::CopyTexture(TexturePtr src, TexturePtr dst) {
    if (!m_CommandList || !src || !dst) return;
    // Implementation for texture copy
}

uint32_t NVRHICommandList::BeginTimestamp() {
    return m_TimestampCounter++;
}

void NVRHICommandList::EndTimestamp(uint32_t queryId) {
    // Implementation for timestamp queries
}

double NVRHICommandList::GetTimestamp(uint32_t queryId) {
    auto it = m_Timestamps.find(queryId);
    if (it != m_Timestamps.end()) {
        return static_cast<double>(it->second);
    }
    return 0.0;
}

// ============================================================================
// NVRHIDevice Implementation
// ============================================================================

NVRHIDevice::NVRHIDevice() {
}

NVRHIDevice::~NVRHIDevice() {
    Shutdown();
}

bool NVRHIDevice::Initialize(GraphicsBackend backend, uint32_t width, uint32_t height, void* windowHandle) {
    m_Backend = backend;
    m_ScreenWidth = width;
    m_ScreenHeight = height;
    m_WindowHandle = windowHandle;

    // Determine which device factory to use based on backend
    nvrhi::DeviceDesc deviceDesc;
    
#ifdef _WIN32
    if (backend == GraphicsBackend::D3D12) {
        // Initialize D3D12 device
        m_Device = nvrhi::createDevice(nvrhi::GraphicsAPI::D3D12);
    } else if (backend == GraphicsBackend::D3D11) {
        // Initialize D3D11 device
        m_Device = nvrhi::createDevice(nvrhi::GraphicsAPI::D3D11);
    }
#endif
    
    if (backend == GraphicsBackend::Vulkan) {
        // Initialize Vulkan device
        m_Device = nvrhi::createDevice(nvrhi::GraphicsAPI::VULKAN);
    }

    if (!m_Device) {
        std::cerr << "Failed to create NVRHI device with backend: " << static_cast<int>(backend) << std::endl;
        return false;
    }

    std::cout << "NVRHI device initialized successfully with backend: " << static_cast<int>(backend) << std::endl;
    return true;
}

void NVRHIDevice::Shutdown() {
    if (m_Device) {
        m_Device = nullptr;
    }
}

void NVRHIDevice::Present(bool vsync) {
    if (m_Device && m_Framebuffer) {
        // Present the framebuffer to the display
    }
}

void NVRHIDevice::Resize(uint32_t width, uint32_t height) {
    m_ScreenWidth = width;
    m_ScreenHeight = height;
    // Handle swapchain resize
}

BufferPtr NVRHIDevice::CreateBuffer(const BufferDesc& desc) {
    if (!m_Device) return nullptr;

    nvrhi::BufferDesc nvrhiDesc;
    nvrhiDesc.byteSize = desc.size;
    nvrhiDesc.debugName = "Buffer";
    nvrhiDesc.cpuAccess = desc.cpuAccess ? nvrhi::CpuAccessMode::Write : nvrhi::CpuAccessMode::None;

    switch (desc.usage) {
        case BufferUsage::ConstantBuffer:
            nvrhiDesc.initialData = desc.initialData;
            break;
        case BufferUsage::VertexBuffer:
            nvrhiDesc.initialData = desc.initialData;
            break;
        case BufferUsage::IndexBuffer:
            nvrhiDesc.initialData = desc.initialData;
            break;
        case BufferUsage::StructuredBuffer:
            nvrhiDesc.initialData = desc.initialData;
            break;
        default:
            break;
    }

    auto nvrhi_buffer = m_Device->createBuffer(nvrhiDesc);
    if (!nvrhi_buffer) {
        std::cerr << "Failed to create NVRHI buffer" << std::endl;
        return nullptr;
    }

    return std::make_shared<NVRHIBuffer>(nvrhi_buffer, desc);
}

TexturePtr NVRHIDevice::CreateTexture(const TextureDesc& desc) {
    if (!m_Device) return nullptr;

    nvrhi::TextureDesc nvrhiDesc;
    nvrhiDesc.width = desc.width;
    nvrhiDesc.height = desc.height;
    nvrhiDesc.depth = desc.depth;
    nvrhiDesc.mipLevels = desc.mipLevels;
    nvrhiDesc.arraySize = desc.arraySize;
    nvrhiDesc.format = ConvertFormat(desc.format);
    nvrhiDesc.debugName = "Texture";

    auto nvrhi_texture = m_Device->createTexture(nvrhiDesc);
    if (!nvrhi_texture) {
        std::cerr << "Failed to create NVRHI texture" << std::endl;
        return nullptr;
    }

    return std::make_shared<NVRHITexture>(nvrhi_texture, desc);
}

ShaderPtr NVRHIDevice::CreateShader(const ShaderDesc& desc) {
    if (!m_Device) return nullptr;

    nvrhi::ShaderDesc nvrhiDesc;
    nvrhiDesc.shaderType = ConvertShaderType(desc.type);
    nvrhiDesc.debugName = "Shader";

    // Handle different shader languages
    switch (desc.language) {
        case ShaderLanguage::HLSL:
            nvrhiDesc.hlslEntry = desc.entryPoint.c_str();
            break;
        case ShaderLanguage::GLSL:
            // GLSL compilation would require GLSL to SPIR-V conversion
            break;
        case ShaderLanguage::SPIR_V:
            // Direct SPIR-V bytecode
            break;
    }

    auto nvrhi_shader = m_Device->createShader(nvrhiDesc, desc.bytecode.data(), desc.bytecode.size());
    if (!nvrhi_shader) {
        std::cerr << "Failed to create NVRHI shader" << std::endl;
        return nullptr;
    }

    return std::make_shared<NVRHIShader>(nvrhi_shader, desc);
}

PipelinePtr NVRHIDevice::CreatePipeline() {
    if (!m_Device) return nullptr;

    nvrhi::GraphicsPipelineDesc pipelineDesc = CreateGraphicsPipelineDesc();
    auto nvrhiPipeline = m_Device->createGraphicsPipeline(pipelineDesc);

    return std::make_shared<NVRHIPipeline>(nvrhiPipeline);
}

CommandListPtr NVRHIDevice::CreateCommandList() {
    if (!m_Device) return nullptr;

    auto cmdList = m_Device->createCommandList();
    if (!cmdList) {
        std::cerr << "Failed to create NVRHI command list" << std::endl;
        return nullptr;
    }

    return std::make_shared<NVRHICommandList>(cmdList, m_Device);
}

void NVRHIDevice::DestroyBuffer(BufferPtr& buffer) {
    buffer.reset();
}

void NVRHIDevice::DestroyTexture(TexturePtr& texture) {
    texture.reset();
}

void NVRHIDevice::DestroyShader(ShaderPtr& shader) {
    shader.reset();
}

void NVRHIDevice::DestroyPipeline(PipelinePtr& pipeline) {
    pipeline.reset();
}

TexturePtr NVRHIDevice::GetBackBuffer() {
    return m_BackBuffer;
}

TexturePtr NVRHIDevice::GetDepthBuffer() {
    return m_DepthBuffer;
}

uint32_t NVRHIDevice::CreateSampler(bool linearFilter, bool clampUVW) {
    if (!m_Device) return 0;

    nvrhi::SamplerDesc samplerDesc;
    samplerDesc.magFilter = linearFilter ? nvrhi::SamplerFilter::Linear : nvrhi::SamplerFilter::Point;
    samplerDesc.minFilter = linearFilter ? nvrhi::SamplerFilter::Linear : nvrhi::SamplerFilter::Point;
    samplerDesc.addressU = clampUVW ? nvrhi::SamplerAddressMode::Clamp : nvrhi::SamplerAddressMode::Wrap;
    samplerDesc.addressV = clampUVW ? nvrhi::SamplerAddressMode::Clamp : nvrhi::SamplerAddressMode::Wrap;
    samplerDesc.addressW = clampUVW ? nvrhi::SamplerAddressMode::Clamp : nvrhi::SamplerAddressMode::Wrap;

    auto sampler = m_Device->createSampler(samplerDesc);
    if (!sampler) {
        std::cerr << "Failed to create NVRHI sampler" << std::endl;
        return 0;
    }

    return 1; // Return a valid sampler handle
}

void NVRHIDevice::DestroySampler(uint32_t samplerHandle) {
    // Sampler cleanup
}

void NVRHIDevice::WaitForIdle() {
    if (m_Device) {
        m_Device->waitForIdle();
    }
}

void NVRHIDevice::Flush() {
    if (m_Device) {
        // Flush pending commands
    }
}

void NVRHIDevice::SetDebugName(BufferPtr buffer, const std::string& name) {
    // Implementation for setting debug names
}

void NVRHIDevice::SetDebugName(TexturePtr texture, const std::string& name) {
    // Implementation for setting debug names
}

void NVRHIDevice::SetDebugName(ShaderPtr shader, const std::string& name) {
    // Implementation for setting debug names
}

void NVRHIDevice::SetDebugName(PipelinePtr pipeline, const std::string& name) {
    // Implementation for setting debug names
}

uint64_t NVRHIDevice::GetGPUMemoryUsed() const {
    // Implementation would query GPU memory usage from NVRHI device
    return 0;
}

uint64_t NVRHIDevice::GetGPUMemoryAvailable() const {
    // Implementation would query available GPU memory from NVRHI device
    return 0;
}

float NVRHIDevice::GetGPUTemperature() const {
    // Implementation would query GPU temperature (if available)
    return 0.0f;
}

void* NVRHIDevice::GetNativeDevice() {
    return m_Device;
}

void* NVRHIDevice::GetNativeContext() {
    return m_Device;
}

nvrhi::Format NVRHIDevice::ConvertFormat(TextureFormat format) const {
    switch (format) {
        case TextureFormat::R8G8B8A8_UNORM:
            return nvrhi::Format::RGBA8_UNORM;
        case TextureFormat::R16G16B16A16_FLOAT:
            return nvrhi::Format::RGBA16_FLOAT;
        case TextureFormat::R32G32B32A32_FLOAT:
            return nvrhi::Format::RGBA32_FLOAT;
        case TextureFormat::R32_FLOAT:
            return nvrhi::Format::R32_FLOAT;
        case TextureFormat::R32G32_FLOAT:
            return nvrhi::Format::RG32_FLOAT;
        case TextureFormat::D32_FLOAT:
            return nvrhi::Format::D32;
        case TextureFormat::D24_UNORM_S8_UINT:
            return nvrhi::Format::D24S8;
        default:
            return nvrhi::Format::RGBA8_UNORM;
    }
}

TextureFormat NVRHIDevice::ConvertFormatFromNVRHI(nvrhi::Format format) const {
    switch (format) {
        case nvrhi::Format::RGBA8_UNORM:
            return TextureFormat::R8G8B8A8_UNORM;
        case nvrhi::Format::RGBA16_FLOAT:
            return TextureFormat::R16G16B16A16_FLOAT;
        case nvrhi::Format::RGBA32_FLOAT:
            return TextureFormat::R32G32B32A32_FLOAT;
        case nvrhi::Format::R32_FLOAT:
            return TextureFormat::R32_FLOAT;
        case nvrhi::Format::RG32_FLOAT:
            return TextureFormat::R32G32_FLOAT;
        case nvrhi::Format::D32:
            return TextureFormat::D32_FLOAT;
        case nvrhi::Format::D24S8:
            return TextureFormat::D24_UNORM_S8_UINT;
        default:
            return TextureFormat::R8G8B8A8_UNORM;
    }
}

nvrhi::ShaderType NVRHIDevice::ConvertShaderType(ShaderType type) const {
    switch (type) {
        case ShaderType::Vertex:
            return nvrhi::ShaderType::Vertex;
        case ShaderType::Fragment:
            return nvrhi::ShaderType::Pixel;
        case ShaderType::Compute:
            return nvrhi::ShaderType::Compute;
        case ShaderType::Geometry:
            return nvrhi::ShaderType::Geometry;
        case ShaderType::TessControl:
            return nvrhi::ShaderType::Hull;
        case ShaderType::TessEval:
            return nvrhi::ShaderType::Domain;
        default:
            return nvrhi::ShaderType::Vertex;
    }
}

nvrhi::GraphicsPipelineDesc NVRHIDevice::CreateGraphicsPipelineDesc() const {
    nvrhi::GraphicsPipelineDesc desc;
    // Default pipeline configuration
    return desc;
}

// ============================================================================
// Global Device Functions
// ============================================================================

DevicePtr GetDevice() {
    return g_Device;
}

void SetDevice(DevicePtr device) {
    g_Device = device;
}

DevicePtr CreateDevice(GraphicsBackend backend) {
    auto device = std::make_shared<NVRHIDevice>();
    SetDevice(device);
    return device;
}

} // namespace Graphics
