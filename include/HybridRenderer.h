#pragma once

#include <memory>
#include <vector>
#include "Shader.h"
#include "Camera.h"
#include "Light.h"
#include "GBuffer.h"
#include "RenderPass.h"
#include "GPUCullingSystem.h"
#include <glm/glm.hpp>

class GameObject;
class PostProcessing;
class ParticleSystem;

/**
 * @class HybridRenderer
 * @brief Deferred + Forward Hybrid Renderer with GPU-driven Culling
 * 
 * Architecture:
 * 1. GPU-driven Culling: Compute shaders determine visibility and LOD
 * 2. Deferred Geometry: Opaque objects → G-Buffer (PBR attributes)
 * 3. Deferred Lighting: Compute lighting in screen-space (PBR with shadows)
 * 4. Forward Rendering: Transparent objects + particles rendered forward
 * 5. Post-Processing: SSAO, SSR, TAA, bloom, volumetric fog
 * 
 * The pipeline uses an SRP-like architecture where each stage is a
 * RenderPass that can be enabled/disabled, reordered, or replaced.
 */
class HybridRenderer {
public:
    static const int MAX_LIGHTS = 32;
    static const int WORKGROUP_SIZE = 32;

    HybridRenderer();
    ~HybridRenderer();

    bool Initialize();
    void Shutdown();

    // Main rendering loop
    void Update(float deltaTime);
    void Render();

    // Scene setup
    void SetCamera(Camera* camera) { m_Camera = camera; }
    void SetSceneRoot(std::shared_ptr<GameObject> root) { m_SceneRoot = root; }

    // Light management
    void AddLight(const Light& light) { m_Lights.push_back(light); }
    void ClearLights() { m_Lights.clear(); }
    std::vector<Light>& GetLights() { return m_Lights; }

    // GPU Culling System
    GPUCullingSystem* GetCullingSystem() { return m_CullingSystem.get(); }
    bool GetGPUCullingEnabled() const { return m_GPUCullingEnabled; }
    void SetGPUCullingEnabled(bool enabled) { m_GPUCullingEnabled = enabled; }

    // Rendering mode control
    enum class RenderMode {
        DeferredOnly,        // Pure deferred rendering
        ForwardOnly,         // Pure forward rendering
        HybridOptimized,     // Hybrid: deferred for opaque, forward for transparent
        HybridDebug          // Hybrid with culling visualization
    };

    void SetRenderMode(RenderMode mode) { m_RenderMode = mode; }
    RenderMode GetRenderMode() const { return m_RenderMode; }

    // Hybrid lighting modes
    enum class LightingMode {
        Deferred,            // All lights in screen-space (G-Buffer limit: 32)
        ForwardPlus,         // Clustered forward+: tiles compute light lists
        TiledDeferred         // Tiled deferred: tiles compute light lists
    };

    void SetLightingMode(LightingMode mode) { m_LightingMode = mode; }
    LightingMode GetLightingMode() const { return m_LightingMode; }

    // Pipeline management
    RenderPipeline* GetPipeline() { return m_Pipeline.get(); }
    void RecreateDefaultPipeline();

    // Debug visualization
    void SetShowCullingBounds(bool show) { m_ShowCullingBounds = show; }
    void SetShowGBuffer(bool show) { m_ShowGBuffer = show; }
    void SetShowLightHeatmap(bool show) { m_ShowLightHeatmap = show; }

private:
    // ===== Pipeline Execution =====
    void ExecuteShadowPass();
    void ExecuteGPUCulling();
    void ExecuteGeometryPass();
    void ExecuteLightingPass();
    void ExecuteTransparentPass();
    void ExecutePostProcessing();
    void ExecuteCompositePass();

    // ===== Rendering Helpers =====
    void CollectRenderableObjects();
    void UpdateLightData();
    void RenderDebugVis();

    // ===== Core State =====
    Camera* m_Camera;
    std::shared_ptr<GameObject> m_SceneRoot;
    std::vector<Light> m_Lights;
    RenderMode m_RenderMode;
    LightingMode m_LightingMode;

    // ===== GPU Resources =====
    std::unique_ptr<GBuffer> m_GBuffer;
    std::unique_ptr<GPUCullingSystem> m_CullingSystem;
    std::unique_ptr<RenderPipeline> m_Pipeline;

    // Shader programs for hybrid rendering
    std::unique_ptr<Shader> m_GeometryShader;           // G-Buffer fill
    std::unique_ptr<Shader> m_DeferredLightingShader;   // Screen-space lighting
    std::unique_ptr<Shader> m_TransparentShader;        // Forward transparent
    std::unique_ptr<Shader> m_CompositeShader;          // Final composition
    std::unique_ptr<Shader> m_ShadowShader;             // Shadow map generation
    std::unique_ptr<Shader> m_DebugVisShader;           // Debug visualization

    // Light data buffer (GPU)
    unsigned int m_LightDataUBO;
    unsigned int m_LightGridSSBO;    // For tiled/clustered lighting
    unsigned int m_LightListSSBO;    // Per-tile light indices

    // Renderable objects
    struct RenderableObject {
        GameObject* gameObject;
        glm::mat4 worldMatrix;
        uint32_t lodLevel;
        bool isVisible;
    };
    std::vector<RenderableObject> m_RenderableObjects;

    // Post-processing
    std::unique_ptr<PostProcessing> m_PostProcessing;
    std::unique_ptr<ParticleSystem> m_ParticleSystem;

    // Debug state
    bool m_ShowCullingBounds;
    bool m_ShowGBuffer;
    bool m_ShowLightHeatmap;
    bool m_GPUCullingEnabled;

    // Framebuffer handles
    unsigned int m_MainFramebuffer;
    unsigned int m_TransparencyFramebuffer;

    // Helper methods
    void SetupDefaultShaders();
    void SetupGPUBuffers();
    void CleanupGPUBuffers();
};
