#pragma once

#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include "Shader.h"

/**
 * @class GPUCullingSystem
 * @brief GPU-driven hierarchical culling using compute shaders
 * 
 * Replaces CPU frustum culling with GPU compute shader-based culling:
 * - Frustum culling: Tests bounding boxes/spheres against camera frustum
 * - Occlusion culling: Tests against depth hierarchy from previous frame
 * - LOD selection: Computes LOD level based on distance and screen coverage
 * 
 * All culling operations run on GPU with the results stored in indirect
 * draw buffers, enabling efficient GPU-driven rendering.
 */
class GPUCullingSystem {
public:
    // Per-instance data sent to GPU
    struct CullData {
        glm::mat4 modelMatrix;
        glm::vec4 boundingSphere;        // xyz = center, w = radius
        glm::vec4 aabbMin;               // Bounding box min
        glm::vec4 aabbMax;               // Bounding box max
        uint32_t meshletCount;           // Number of meshlets for this object
        uint32_t lodLevel;               // Current LOD level (populated by compute)
        uint32_t isVisible;              // Visibility flag (populated by compute)
        uint32_t screenCoverage;         // Approximate screen coverage percentage
    };

    // Indirect draw command buffer format
    struct IndirectDrawCommand {
        uint32_t indexCount;
        uint32_t instanceCount;
        uint32_t firstIndex;
        int32_t baseVertex;
        uint32_t baseInstance;
    };

    // Result of culling operations
    struct CullingResults {
        uint32_t visibleCount;
        uint32_t culledCount;
        std::vector<uint32_t> visibleIndices;
        std::vector<uint32_t> lodLevels;
    };

    GPUCullingSystem();
    ~GPUCullingSystem();

    bool Initialize();
    void Shutdown();

    // Setup: Call once per frame before culling
    void SetupCulling(
        const glm::mat4& viewMatrix,
        const glm::mat4& projectionMatrix,
        const glm::vec3& cameraPos,
        size_t instanceCount
    );

    // Culling operations
    void ExecuteFrustumCulling();
    void ExecuteOcclusionCulling(unsigned int depthTexture);
    void ExecuteLODSelection();
    
    // Multi-pass culling (frustum -> occlusion -> LOD)
    void ExecuteAll(unsigned int depthTexture);

    // Data management
    void SetCullData(const std::vector<CullData>& data);
    void SetCullData(const std::vector<CullData>& data, size_t offset, size_t count);
    
    CullingResults GetResults();
    std::vector<IndirectDrawCommand> GetIndirectCommands();

    // GPU buffers
    unsigned int GetCullDataSSBO() const { return m_CullDataSSBO; }
    unsigned int GetVisibilitySSBO() const { return m_VisibilitySSBO; }
    unsigned int GetIndirectCommandBuffer() const { return m_IndirectCommandBuffer; }
    unsigned int GetCounterBuffer() const { return m_CounterBuffer; }

    // Configuration
    void SetMaxInstances(size_t maxInstances);
    void SetFrustumCullingEnabled(bool enabled) { m_FrustumCullingEnabled = enabled; }
    void SetOcclusionCullingEnabled(bool enabled) { m_OcclusionCullingEnabled = enabled; }
    void SetLODSelectionEnabled(bool enabled) { m_LODSelectionEnabled = enabled; }

    // Debug/profiling
    void SetDebugMode(bool enabled) { m_DebugMode = enabled; }
    uint32_t GetVisibleInstanceCount() const { return m_LastVisibleCount; }
    float GetLastComputeTime() const { return m_LastComputeTime; }

private:
    // Shader programs
    std::unique_ptr<Shader> m_FrustumCullingShader;
    std::unique_ptr<Shader> m_OcclusionCullingShader;
    std::unique_ptr<Shader> m_LODSelectionShader;
    std::unique_ptr<Shader> m_IndirectCommandBuilderShader;

    // GPU storage buffers
    unsigned int m_CullDataSSBO;              // Input: Instance cull data
    unsigned int m_VisibilitySSBO;            // Output: Visibility flags
    unsigned int m_IndirectCommandBuffer;     // Output: GPU draw commands
    unsigned int m_CounterBuffer;             // Atomic counter for visible instances
    unsigned int m_LODLevelSSBO;              // Output: LOD levels per instance
    unsigned int m_CullingConstantsUBO;       // Frustum planes + camera data

    // Configuration
    size_t m_MaxInstances;
    bool m_FrustumCullingEnabled;
    bool m_OcclusionCullingEnabled;
    bool m_LODSelectionEnabled;
    bool m_DebugMode;

    // Cached state
    glm::mat4 m_ViewMatrix;
    glm::mat4 m_ProjectionMatrix;
    glm::vec3 m_CameraPosition;
    uint32_t m_LastVisibleCount;
    float m_LastComputeTime;

    // Helper methods
    void ExtractFrustumPlanes(const glm::mat4& viewProj, std::vector<glm::vec4>& planes);
    void UploadFrustumPlanes(const std::vector<glm::vec4>& planes);
};
