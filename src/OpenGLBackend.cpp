#include "OpenGLBackend.h"
#include "GLExtensions.h"
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>
#include <chrono>

// Compatibility flags for GLExtensions.h
#define GLAD_GL_ARB_tessellation_shader false
#define GLAD_GL_ARB_geometry_shader4 false
#define GLAD_GL_ARB_compute_shader (glDispatchCompute != nullptr)
#define GLAD_GL_ARB_compute_variable_group_size false

OpenGLBackend::OpenGLBackend() = default;

OpenGLBackend::~OpenGLBackend() {
    Shutdown();
}

bool OpenGLBackend::Init(uint32_t width, uint32_t height, void* windowHandle) {
    m_Width = width;
    m_Height = height;
    SPDLOG_INFO("OpenGL backend initialized: {}x{}", width, height);
    return true;
}

void OpenGLBackend::Shutdown() {
    for (auto& query : m_QueryObjects) {
        glDeleteQueries(1, &query.second);
    }
    m_QueryObjects.clear();
    m_QueryStartTimes.clear();
    SPDLOG_INFO("OpenGL backend shutdown");
}

std::string OpenGLBackend::GetAPIName() const {
    return "OpenGL 3.3+";
}

GPUDeviceInfo OpenGLBackend::GetDeviceInfo(uint32_t deviceIndex) const {
    GPUDeviceInfo info;
    info.id = 0;
    info.name = (const char*)glGetString(GL_RENDERER);
    info.driverVersion = (const char*)glGetString(GL_VERSION);
    
    GLint maxMemory = 0;
    glGetIntegerv(GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX, &maxMemory);
    info.dedicatedMemory = maxMemory * 1024; // Convert KB to bytes
    
    return info;
}

bool OpenGLBackend::SupportsFeature(const std::string& featureName) const {
    // Check for OpenGL extensions
    if (featureName == "tessellation") return GLAD_GL_ARB_tessellation_shader;
    if (featureName == "geometry") return GLAD_GL_ARB_geometry_shader4;
    if (featureName == "compute") return GLAD_GL_ARB_compute_shader;
    if (featureName == "indirect_dispatch") return GLAD_GL_ARB_compute_variable_group_size;
    return false;
}

uint32_t OpenGLBackend::GetMaxTextureSize() const {
    GLint maxSize = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxSize);
    return maxSize;
}

glm::uvec2 OpenGLBackend::GetMaxFramebufferSize() const {
    GLint maxWidth = 0, maxHeight = 0;
    glGetIntegerv(GL_MAX_FRAMEBUFFER_WIDTH, &maxWidth);
    glGetIntegerv(GL_MAX_FRAMEBUFFER_HEIGHT, &maxHeight);
    return glm::uvec2(maxWidth, maxHeight);
}

std::shared_ptr<RenderResource> OpenGLBackend::CreateBuffer(
    size_t size,
    const void* data,
    uint32_t usageFlags)
{
    GLuint buffer = 0;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_COPY_WRITE_BUFFER, buffer);
    glBufferData(GL_COPY_WRITE_BUFFER, size, data, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);

    auto resource = std::make_shared<RenderResource>(RenderResource::Type::Buffer);
    resource->nativeHandle = buffer;
    return resource;
}

std::shared_ptr<RenderResource> OpenGLBackend::CreateTexture(
    uint32_t width,
    uint32_t height,
    const std::string& format,
    const void* data,
    bool generateMips)
{
    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    uint32_t glFormat = GetGLFormat(format);
    glTexImage2D(GL_TEXTURE_2D, 0, glFormat, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

    if (generateMips) {
        glGenerateMipmap(GL_TEXTURE_2D);
    }

    glBindTexture(GL_TEXTURE_2D, 0);

    auto resource = std::make_shared<RenderResource>(RenderResource::Type::Texture);
    resource->nativeHandle = texture;
    return resource;
}

std::shared_ptr<RenderResource> OpenGLBackend::CreateTexture3D(
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    const std::string& format,
    const void* data)
{
    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_3D, texture);

    uint32_t glFormat = GetGLFormat(format);
    glTexImage3D(GL_TEXTURE_3D, 0, glFormat, width, height, depth, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

    glBindTexture(GL_TEXTURE_3D, 0);

    auto resource = std::make_shared<RenderResource>(RenderResource::Type::Texture);
    resource->nativeHandle = texture;
    return resource;
}

std::shared_ptr<RenderResource> OpenGLBackend::CreateCubemap(
    uint32_t size,
    const std::string& format,
    const std::vector<const void*>& faceImages)
{
    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture);

    uint32_t glFormat = GetGLFormat(format);
    const GLenum faces[] = {
        GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
    };

    for (int i = 0; i < 6; i++) {
        const void* faceData = (i < faceImages.size()) ? faceImages[i] : nullptr;
        glTexImage2D(faces[i], 0, glFormat, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, faceData);
    }

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    auto resource = std::make_shared<RenderResource>(RenderResource::Type::Texture);
    resource->nativeHandle = texture;
    return resource;
}

std::shared_ptr<RenderResource> OpenGLBackend::CreateFramebuffer(
    uint32_t width,
    uint32_t height,
    const std::vector<std::string>& colorFormats,
    const std::string& depthFormat)
{
    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // Attach color textures
    for (size_t i = 0; i < colorFormats.size(); i++) {
        GLuint colorTexture = 0;
        glGenTextures(1, &colorTexture);
        glBindTexture(GL_TEXTURE_2D, colorTexture);
        
        uint32_t glFormat = GetGLFormat(colorFormats[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, glFormat, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colorTexture, 0);
    }

    // Attach depth texture if specified
    if (!depthFormat.empty()) {
        GLuint depthTexture = 0;
        glGenTextures(1, &depthTexture);
        glBindTexture(GL_TEXTURE_2D, depthTexture);
        
        uint32_t glFormat = GetGLFormat(depthFormat);
        glTexImage2D(GL_TEXTURE_2D, 0, glFormat, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture, 0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    auto resource = std::make_shared<RenderResource>(RenderResource::Type::Framebuffer);
    resource->nativeHandle = fbo;
    return resource;
}

std::shared_ptr<RenderResource> OpenGLBackend::CreateShader(
    const std::string& source,
    ShaderType type)
{
    GLenum glType = GL_VERTEX_SHADER;
    switch (type) {
        case ShaderType::Vertex: glType = GL_VERTEX_SHADER; break;
        case ShaderType::Fragment: glType = GL_FRAGMENT_SHADER; break;
        case ShaderType::Compute: glType = GL_COMPUTE_SHADER; break;
        case ShaderType::Geometry: glType = GL_GEOMETRY_SHADER; break;
        case ShaderType::TessControl: glType = GL_TESS_CONTROL_SHADER; break;
        case ShaderType::TessEval: glType = GL_TESS_EVALUATION_SHADER; break;
        default: break;
    }

    GLuint shader = glCreateShader(glType);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    // Check compilation errors
    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        SPDLOG_ERROR("Shader compilation error: {}", infoLog);
    }

    auto resource = std::make_shared<RenderResource>(RenderResource::Type::Pipeline);
    resource->nativeHandle = shader;
    return resource;
}

std::shared_ptr<RenderResource> OpenGLBackend::CreatePipeline(const void* config) {
    // Placeholder - pipeline creation is simplified in OpenGL
    auto resource = std::make_shared<RenderResource>(RenderResource::Type::Pipeline);
    return resource;
}

void OpenGLBackend::UpdateBuffer(
    const std::shared_ptr<RenderResource>& buffer,
    size_t offset,
    size_t size,
    const void* data)
{
    GLuint buf = static_cast<GLuint>(buffer->nativeHandle);
    glBindBuffer(GL_COPY_WRITE_BUFFER, buf);
    glBufferSubData(GL_COPY_WRITE_BUFFER, offset, size, data);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

void OpenGLBackend::UpdateTexture(
    const std::shared_ptr<RenderResource>& texture,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height,
    const void* data)
{
    GLuint tex = static_cast<GLuint>(texture->nativeHandle);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void OpenGLBackend::CopyBuffer(
    const std::shared_ptr<RenderResource>& src,
    const std::shared_ptr<RenderResource>& dst,
    size_t size,
    size_t srcOffset,
    size_t dstOffset)
{
    GLuint srcBuf = static_cast<GLuint>(src->nativeHandle);
    GLuint dstBuf = static_cast<GLuint>(dst->nativeHandle);
    glBindBuffer(GL_COPY_READ_BUFFER, srcBuf);
    glBindBuffer(GL_COPY_WRITE_BUFFER, dstBuf);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, srcOffset, dstOffset, size);
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

void OpenGLBackend::CopyBufferToTexture(
    const std::shared_ptr<RenderResource>& buffer,
    const std::shared_ptr<RenderResource>& texture,
    uint32_t width,
    uint32_t height)
{
    GLuint buf = static_cast<GLuint>(buffer->nativeHandle);
    GLuint tex = static_cast<GLuint>(texture->nativeHandle);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void OpenGLBackend::DestroyResource(const std::shared_ptr<RenderResource>& resource) {
    if (!resource) return;

    GLuint handle = static_cast<GLuint>(resource->nativeHandle);
    switch (resource->type) {
        case RenderResource::Type::Buffer:
            glDeleteBuffers(1, &handle);
            break;
        case RenderResource::Type::Texture:
            glDeleteTextures(1, &handle);
            break;
        case RenderResource::Type::Framebuffer:
            glDeleteFramebuffers(1, &handle);
            break;
        case RenderResource::Type::Pipeline:
            glDeleteShader(handle);
            break;
        default:
            break;
    }
}

void OpenGLBackend::WaitForGPU() {
    glFinish();
}

void OpenGLBackend::BeginCommandBuffer() {
    // OpenGL doesn't use explicit command buffers
}

void OpenGLBackend::EndCommandBuffer() {
    // OpenGL doesn't use explicit command buffers
}

void OpenGLBackend::BeginRenderPass(
    const std::shared_ptr<RenderResource>& framebuffer,
    const glm::vec4& clearColor,
    float clearDepth)
{
    GLuint fbo = static_cast<GLuint>(framebuffer->nativeHandle);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void OpenGLBackend::EndRenderPass() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void OpenGLBackend::SetViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
    glViewport(x, y, width, height);
}

void OpenGLBackend::BindPipeline(const std::shared_ptr<RenderResource>& pipeline) {
    GLuint program = static_cast<GLuint>(pipeline->nativeHandle);
    glUseProgram(program);
    m_CurrentProgram = program;
}

void OpenGLBackend::BindVertexBuffer(
    const std::shared_ptr<RenderResource>& buffer,
    size_t offset)
{
    GLuint buf = static_cast<GLuint>(buffer->nativeHandle);
    glBindBuffer(GL_ARRAY_BUFFER, buf);
}

void OpenGLBackend::BindIndexBuffer(
    const std::shared_ptr<RenderResource>& buffer,
    size_t offset)
{
    GLuint buf = static_cast<GLuint>(buffer->nativeHandle);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buf);
}

void OpenGLBackend::BindTexture(
    uint32_t slot,
    const std::shared_ptr<RenderResource>& texture,
    const std::shared_ptr<RenderResource>& sampler)
{
    glActiveTexture(GL_TEXTURE0 + slot);
    GLuint tex = static_cast<GLuint>(texture->nativeHandle);
    glBindTexture(GL_TEXTURE_2D, tex);
}

void OpenGLBackend::BindStorageBuffer(
    uint32_t slot,
    const std::shared_ptr<RenderResource>& buffer)
{
    GLuint buf = static_cast<GLuint>(buffer->nativeHandle);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, slot, buf);
}

void OpenGLBackend::SetPushConstants(
    const void* data,
    size_t size,
    size_t offset)
{
    // OpenGL doesn't have push constants - use UBOs or SSBOs instead
}

void OpenGLBackend::Draw(
    uint32_t vertexCount,
    uint32_t instanceCount,
    uint32_t firstVertex,
    uint32_t firstInstance)
{
    glDrawArraysInstanced(GL_TRIANGLES, firstVertex, vertexCount, instanceCount);
}

void OpenGLBackend::DrawIndexed(
    uint32_t indexCount,
    uint32_t instanceCount,
    uint32_t firstIndex,
    int32_t vertexOffset,
    uint32_t firstInstance)
{
    glDrawElementsInstancedBaseVertexBaseInstance(
        GL_TRIANGLES, indexCount, GL_UNSIGNED_INT,
        (void*)(firstIndex * sizeof(uint32_t)), instanceCount,
        vertexOffset, firstInstance);
}

void OpenGLBackend::DrawIndirect(
    const std::shared_ptr<RenderResource>& indirectBuffer,
    size_t offset,
    uint32_t drawCount)
{
    GLuint buf = static_cast<GLuint>(indirectBuffer->nativeHandle);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, buf);
    glMultiDrawArraysIndirect(GL_TRIANGLES, (void*)offset, drawCount, sizeof(DrawArraysIndirectCommand));
}

void OpenGLBackend::Dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
    glDispatchCompute(groupCountX, groupCountY, groupCountZ);
}

void OpenGLBackend::DispatchIndirect(
    const std::shared_ptr<RenderResource>& indirectBuffer,
    size_t offset)
{
    GLuint buf = static_cast<GLuint>(indirectBuffer->nativeHandle);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, buf);
    glDispatchComputeIndirect(offset);
}

void OpenGLBackend::GPUMemoryBarrier(uint32_t barrierType) {
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}

void OpenGLBackend::FramebufferBarrier() {
    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
}

uint32_t OpenGLBackend::GetFrameIndex() const {
    return m_FrameIndex;
}

uint64_t OpenGLBackend::GetDeviceHandle(uint32_t deviceIndex) const {
    return 0; // OpenGL single GPU
}

void OpenGLBackend::BeginGPUQuery(const std::string& label) {
    if (m_QueryObjects.find(label) == m_QueryObjects.end()) {
        GLuint query = 0;
        glGenQueries(1, &query);
        m_QueryObjects[label] = query;
    }
    m_QueryStartTimes[label] = std::chrono::high_resolution_clock::now();
    glBeginQuery(GL_TIME_ELAPSED, m_QueryObjects[label]);
}

double OpenGLBackend::EndGPUQuery(const std::string& label) {
    glEndQuery(GL_TIME_ELAPSED);

    GLuint64 elapsedTime = 0;
    glGetQueryObjectui64v(m_QueryObjects[label], GL_QUERY_RESULT_AVAILABLE, &elapsedTime);

    if (elapsedTime) {
        glGetQueryObjectui64v(m_QueryObjects[label], GL_QUERY_RESULT, &elapsedTime);
        return elapsedTime / 1e6; // Convert nanoseconds to milliseconds
    }
    return 0.0;
}

float OpenGLBackend::GetGPUUtilization(uint32_t deviceIndex) const {
    return 0.0f; // Not available in OpenGL
}

uint64_t OpenGLBackend::GetGPUMemoryUsage(uint32_t deviceIndex) const {
    GLint memoryUsage = 0;
    glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &memoryUsage);
    return memoryUsage * 1024; // Convert KB to bytes
}

uint64_t OpenGLBackend::GetGPUMemoryTotal(uint32_t deviceIndex) const {
    GLint totalMemory = 0;
    glGetIntegerv(GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX, &totalMemory);
    return totalMemory * 1024; // Convert KB to bytes
}

uint32_t OpenGLBackend::GetGLFormat(const std::string& format) const {
    if (format == "RGBA8") return GL_RGBA8;
    if (format == "RGB8") return GL_RGB8;
    if (format == "RGBA32F") return GL_RGBA32F;
    if (format == "RGB32F") return GL_RGB32F;
    if (format == "D32F") return GL_DEPTH_COMPONENT32F;
    if (format == "D24S8") return GL_DEPTH24_STENCIL8;
    return GL_RGBA8;
}

uint32_t OpenGLBackend::GetGLUsageFlags(uint32_t usageFlags) const {
    return GL_DYNAMIC_DRAW;
}

