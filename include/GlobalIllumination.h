#pragma once

#include <memory>
#include <vector>
#include "Camera.h"
#include "Light.h"
#include "Shader.h"

class VoxelGrid;
class LightPropagationVolume;

/**
 * @class GlobalIllumination
 * @brief Manages Global Illumination rendering with multiple techniques
 * 
 * Supports three GI techniques:
 * - VCT (Voxel Cone Tracing): High-quality dynamic GI
 * - LPV (Light Propagation Volumes): Fast grid-based GI
 * - SSGI (Screen Space Global Illumination): Screen-space fallback
 * - Hybrid: Combines VCT + SSGI for best quality
 */
class GlobalIllumination {
public:
    enum class Technique {
        None = 0,
        VCT = 1,        // Voxel Cone Tracing
        LPV = 2,        // Light Propagation Volumes
        SSGI = 3,       // Screen Space GI
        Hybrid = 4,     // VCT + SSGI
        Probes = 5,     // Probe-only
        ProbesVCT = 6   // Probes + VCT hybrid
    };
    
    enum class Quality {
        Low = 0,        // 64³ voxels, 3 cones, no temporal
        Medium = 1,     // 128³ voxels, 5 cones, basic temporal
        High = 2,       // 256³ voxels, 5 cones, full temporal
        Ultra = 3       // 512³ voxels, 9 cones, advanced temporal
    };

    GlobalIllumination();
    ~GlobalIllumination();

    // Initialization
    bool Initialize(int screenWidth, int screenHeight);
    void Shutdown();

    // Main rendering
    void Update(float deltaTime);
    void Render(Camera* camera, const std::vector<Light>& lights, 
                const std::vector<class GameObject*>& objects);

    // Configuration
    void SetTechnique(Technique technique) { m_Technique = technique; }
    Technique GetTechnique() const { return m_Technique; }
    
    void SetQuality(Quality quality);
    Quality GetQuality() const { return m_Quality; }
    
    void SetEnabled(bool enabled) { m_Enabled = enabled; }
    bool IsEnabled() const { return m_Enabled; }
    
    void SetIntensity(float intensity) { m_Intensity = intensity; }
    float GetIntensity() const { return m_Intensity; }

    // Getters for integration with lighting pass
    unsigned int GetGITexture() const { return m_GITexture; }
    VoxelGrid* GetVoxelGrid() const { return m_VoxelGrid.get(); }
    
    // Probe management
    class ProbeGrid* GetProbeGrid() const { return m_ProbeGrid.get(); }
    void SetProbeBlendWeight(float weight) { m_ProbeBlendWeight = weight; }
    float GetProbeBlendWeight() const { return m_ProbeBlendWeight; }
    
    // Debug visualization
    void SetShowVoxels(bool show) { m_ShowVoxels = show; }
    bool GetShowVoxels() const { return m_ShowVoxels; }
    void RenderDebugVisualization(Camera* camera);

private:
    // Rendering passes
    void RenderVCT(Camera* camera, const std::vector<Light>& lights,
                   const std::vector<GameObject*>& objects);
    void RenderLPV(Camera* camera, const std::vector<Light>& lights);
    void RenderSSGI(Camera* camera);
    void RenderHybrid(Camera* camera, const std::vector<Light>& lights,
                      const std::vector<GameObject*>& objects);

    // Helper methods
    void UpdateQualitySettings();
    void CreateGITexture(int width, int height);
    void CreateSSGIResources(int width, int height);

    // Configuration
    Technique m_Technique;
    Quality m_Quality;
    bool m_Enabled;
    float m_Intensity;
    bool m_ShowVoxels;

    // Screen dimensions
    int m_ScreenWidth;
    int m_ScreenHeight;

    // GI output texture (RGBA16F)
    unsigned int m_GITexture;
    unsigned int m_GIFramebuffer;

    // Technique-specific systems
    std::unique_ptr<VoxelGrid> m_VoxelGrid;
    std::unique_ptr<LightPropagationVolume> m_LPV;
    std::unique_ptr<class ProbeGrid> m_ProbeGrid;
    float m_ProbeBlendWeight;

    // SSGI resources
    unsigned int m_SSGITexture;
    unsigned int m_SSGIBlurTexture;
    std::unique_ptr<Shader> m_SSGIShader;
    std::unique_ptr<Shader> m_SSGIBlurShader;

    // Cone tracing shader
    std::unique_ptr<Shader> m_ConeTraceShader;

    // Temporal accumulation
    unsigned int m_TemporalTexture[2];  // Ping-pong buffers
    int m_CurrentTemporalIndex;
    std::unique_ptr<Shader> m_TemporalShader;

    // Quality settings (updated by SetQuality)
    int m_VoxelResolution;
    int m_NumDiffuseCones;
    bool m_UseTemporalFiltering;
    float m_TemporalBlendFactor;

    // Debug visualization
    std::unique_ptr<Shader> m_VoxelDebugShader;
    unsigned int m_DebugVAO;
    unsigned int m_DebugVBO;
};
