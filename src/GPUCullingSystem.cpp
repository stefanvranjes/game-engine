#include "GPUCullingSystem.h"
#include "GLExtensions.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <cmath>

GPUCullingSystem::GPUCullingSystem()
    : m_CullDataSSBO(0), m_VisibilitySSBO(0), m_IndirectCommandBuffer(0),
      m_CounterBuffer(0), m_LODLevelSSBO(0), m_CullingConstantsUBO(0),
      m_MaxInstances(8192), m_FrustumCullingEnabled(true),
      m_OcclusionCullingEnabled(true), m_LODSelectionEnabled(true),
      m_DebugMode(false), m_LastVisibleCount(0), m_LastComputeTime(0.0f) {
}

GPUCullingSystem::~GPUCullingSystem() {
    Shutdown();
}

bool GPUCullingSystem::Initialize() {
    // Create shader programs
    m_FrustumCullingShader = std::make_unique<Shader>();
    if (!m_FrustumCullingShader->LoadComputeShader("shaders/gpu_cull_frustum.comp")) {
        return false;
    }

    m_OcclusionCullingShader = std::make_unique<Shader>();
    if (!m_OcclusionCullingShader->LoadComputeShader("shaders/gpu_cull_occlusion.comp")) {
        return false;
    }

    // Create GPU storage buffers
    glGenBuffers(1, &m_CullDataSSBO);
    glGenBuffers(1, &m_VisibilitySSBO);
    glGenBuffers(1, &m_IndirectCommandBuffer);
    glGenBuffers(1, &m_CounterBuffer);
    glGenBuffers(1, &m_LODLevelSSBO);
    glGenBuffers(1, &m_CullingConstantsUBO);

    // Allocate buffer storage
    size_t cullDataSize = m_MaxInstances * sizeof(CullData);
    glBindBuffer(GL_COPY_READ_BUFFER, m_CullDataSSBO);
    glBufferStorage(GL_COPY_READ_BUFFER, cullDataSize, nullptr, GL_DYNAMIC_STORAGE_BIT);

    size_t visibilitySize = m_MaxInstances * sizeof(uint32_t);
    glBindBuffer(GL_COPY_WRITE_BUFFER, m_VisibilitySSBO);
    glBufferStorage(GL_COPY_WRITE_BUFFER, visibilitySize, nullptr, GL_DYNAMIC_STORAGE_BIT);

    size_t indirectSize = m_MaxInstances * sizeof(IndirectDrawCommand);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, m_IndirectCommandBuffer);
    glBufferStorage(GL_DRAW_INDIRECT_BUFFER, indirectSize, nullptr, GL_DYNAMIC_STORAGE_BIT);

    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_CounterBuffer);
    glBufferStorage(GL_ATOMIC_COUNTER_BUFFER, sizeof(uint32_t), nullptr, GL_DYNAMIC_STORAGE_BIT);

    glBindBuffer(GL_COPY_WRITE_BUFFER, m_LODLevelSSBO);
    glBufferStorage(GL_COPY_WRITE_BUFFER, m_MaxInstances * sizeof(uint32_t), nullptr, GL_DYNAMIC_STORAGE_BIT);

    glBindBuffer(GL_UNIFORM_BUFFER, m_CullingConstantsUBO);
    glBufferStorage(GL_UNIFORM_BUFFER, 
        sizeof(glm::vec4) * 6 +      // 6 frustum planes
        sizeof(glm::mat4) * 2 +      // view and projection matrices
        sizeof(glm::vec3) + sizeof(float) * 4, // camera data
        nullptr, GL_DYNAMIC_STORAGE_BIT);

    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);

    return true;
}

void GPUCullingSystem::Shutdown() {
    if (m_CullDataSSBO) glDeleteBuffers(1, &m_CullDataSSBO);
    if (m_VisibilitySSBO) glDeleteBuffers(1, &m_VisibilitySSBO);
    if (m_IndirectCommandBuffer) glDeleteBuffers(1, &m_IndirectCommandBuffer);
    if (m_CounterBuffer) glDeleteBuffers(1, &m_CounterBuffer);
    if (m_LODLevelSSBO) glDeleteBuffers(1, &m_LODLevelSSBO);
    if (m_CullingConstantsUBO) glDeleteBuffers(1, &m_CullingConstantsUBO);

    m_FrustumCullingShader.reset();
    m_OcclusionCullingShader.reset();
}

void GPUCullingSystem::SetupCulling(
    const glm::mat4& viewMatrix,
    const glm::mat4& projectionMatrix,
    const glm::vec3& cameraPos,
    size_t instanceCount) {

    m_ViewMatrix = viewMatrix;
    m_ProjectionMatrix = projectionMatrix;
    m_CameraPosition = cameraPos;

    // Extract frustum planes from view-projection matrix
    glm::mat4 viewProj = projectionMatrix * viewMatrix;
    std::vector<glm::vec4> planes;
    ExtractFrustumPlanes(viewProj, planes);

    // Upload frustum planes and camera data to UBO
    UploadFrustumPlanes(planes);

    // Reset counter buffer
    uint32_t zero = 0;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_CounterBuffer);
    glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(uint32_t), &zero);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
}

void GPUCullingSystem::SetCullData(const std::vector<CullData>& data) {
    SetCullData(data, 0, data.size());
}

void GPUCullingSystem::SetCullData(const std::vector<CullData>& data, size_t offset, size_t count) {
    size_t totalSize = count * sizeof(CullData);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_CullDataSSBO);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset * sizeof(CullData), totalSize, data.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void GPUCullingSystem::ExecuteFrustumCulling() {
    if (!m_FrustumCullingEnabled || !m_FrustumCullingShader) return;

    m_FrustumCullingShader->Use();

    // Bind SSBOs
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_CullDataSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_VisibilitySSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_LODLevelSSBO);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, m_CullingConstantsUBO);

    // Dispatch compute shader
    uint32_t groupCount = (m_MaxInstances + 31) / 32;
    glDispatchCompute(groupCount, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void GPUCullingSystem::ExecuteOcclusionCulling(unsigned int depthTexture) {
    if (!m_OcclusionCullingEnabled || !m_OcclusionCullingShader) return;

    m_OcclusionCullingShader->Use();

    // Bind SSBOs and depth texture
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_CullDataSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_VisibilitySSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_VisibilitySSBO);  // Output to same buffer
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, m_CullingConstantsUBO);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    m_OcclusionCullingShader->SetInt("u_DepthPyramid", 0);

    uint32_t groupCount = (m_MaxInstances + 31) / 32;
    glDispatchCompute(groupCount, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void GPUCullingSystem::ExecuteLODSelection() {
    // LOD selection is computed during frustum culling in the same shader
    // This is a placeholder for potential separate LOD refinement
}

void GPUCullingSystem::ExecuteAll(unsigned int depthTexture) {
    ExecuteFrustumCulling();
    if (m_OcclusionCullingEnabled) {
        ExecuteOcclusionCulling(depthTexture);
    }
    ExecuteLODSelection();
}

GPUCullingSystem::CullingResults GPUCullingSystem::GetResults() {
    CullingResults results;

    // Read visibility buffer from GPU
    glBindBuffer(GL_COPY_READ_BUFFER, m_VisibilitySSBO);
    std::vector<uint32_t> visibilityData(m_MaxInstances);
    glGetBufferSubData(GL_COPY_READ_BUFFER, 0, m_MaxInstances * sizeof(uint32_t), visibilityData.data());

    results.visibleCount = 0;
    results.culledCount = 0;

    for (size_t i = 0; i < m_MaxInstances; ++i) {
        if (visibilityData[i]) {
            results.visibleIndices.push_back(i);
            results.visibleCount++;
        } else {
            results.culledCount++;
        }
    }

    // Read LOD levels
    glBindBuffer(GL_COPY_READ_BUFFER, m_LODLevelSSBO);
    results.lodLevels.resize(m_MaxInstances);
    glGetBufferSubData(GL_COPY_READ_BUFFER, 0, m_MaxInstances * sizeof(uint32_t), results.lodLevels.data());

    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    m_LastVisibleCount = results.visibleCount;

    return results;
}

std::vector<GPUCullingSystem::IndirectDrawCommand> GPUCullingSystem::GetIndirectCommands() {
    std::vector<IndirectDrawCommand> commands;
    
    glBindBuffer(GL_COPY_READ_BUFFER, m_IndirectCommandBuffer);
    std::vector<IndirectDrawCommand> data(m_MaxInstances);
    glGetBufferSubData(GL_COPY_READ_BUFFER, 0, m_MaxInstances * sizeof(IndirectDrawCommand), data.data());
    glBindBuffer(GL_COPY_READ_BUFFER, 0);

    // Filter out empty commands
    for (const auto& cmd : data) {
        if (cmd.instanceCount > 0) {
            commands.push_back(cmd);
        }
    }

    return commands;
}

void GPUCullingSystem::SetMaxInstances(size_t maxInstances) {
    m_MaxInstances = maxInstances;
    Shutdown();
    Initialize();
}

void GPUCullingSystem::ExtractFrustumPlanes(const glm::mat4& viewProj, std::vector<glm::vec4>& planes) {
    planes.clear();

    // Extract frustum planes from view-projection matrix
    // Based on: https://www.gamedev.net/forums/topic/512123-extracting-view-frustum-planes-from-a-matrix/
    
    for (int i = 0; i < 3; ++i) {
        // Right plane
        glm::vec4 plane = glm::vec4(
            viewProj[0][3] - viewProj[0][i],
            viewProj[1][3] - viewProj[1][i],
            viewProj[2][3] - viewProj[2][i],
            viewProj[3][3] - viewProj[3][i]
        );
        planes.push_back(glm::normalize(plane));

        // Left plane
        plane = glm::vec4(
            viewProj[0][3] + viewProj[0][i],
            viewProj[1][3] + viewProj[1][i],
            viewProj[2][3] + viewProj[2][i],
            viewProj[3][3] + viewProj[3][i]
        );
        planes.push_back(glm::normalize(plane));
    }

    // Ensure we have exactly 6 planes
    while (planes.size() < 6) {
        planes.push_back(glm::vec4(0, 1, 0, 0));
    }
    planes.resize(6);
}

void GPUCullingSystem::UploadFrustumPlanes(const std::vector<glm::vec4>& planes) {
    glBindBuffer(GL_UNIFORM_BUFFER, m_CullingConstantsUBO);

    // Upload frustum planes
    for (size_t i = 0; i < 6 && i < planes.size(); ++i) {
        glBufferSubData(GL_UNIFORM_BUFFER, i * sizeof(glm::vec4), sizeof(glm::vec4), 
                       glm::value_ptr(planes[i]));
    }

    // Upload view and projection matrices
    size_t offset = 6 * sizeof(glm::vec4);
    glBufferSubData(GL_UNIFORM_BUFFER, offset, sizeof(glm::mat4), glm::value_ptr(m_ViewMatrix));
    offset += sizeof(glm::mat4);
    glBufferSubData(GL_UNIFORM_BUFFER, offset, sizeof(glm::mat4), glm::value_ptr(m_ProjectionMatrix));
    offset += sizeof(glm::mat4);

    // Upload camera position and near/far planes
    glBufferSubData(GL_UNIFORM_BUFFER, offset, sizeof(glm::vec3), glm::value_ptr(m_CameraPosition));

    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}
