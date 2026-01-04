#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "Shader.h"

// Dynamic Diffuse Global Illumination Volume
// Manages real-time probe updates using raytracing and hysteresis.
class DDGIVolume {
public:
    struct Settings {
        glm::ivec3 gridDimensions = glm::ivec3(16, 4, 16);
        glm::vec3 startPosition = glm::vec3(-20.0f, 0.0f, -20.0f);
        glm::vec3 probeSpacing = glm::vec3(4.0f, 4.0f, 4.0f);
        
        int raysPerProbe = 128; // Rays per frame per probe
        float hysteresis = 0.97f; // Temporal blend factor (0.0 = instant, 1.0 = frozen)
        float maxRayDistance = 100.0f;
        int irradianceRes = 8; // Texels per probe (side length, excluding border)
        int distanceRes = 16;  // Texels per probe for visibility
    };

    DDGIVolume();
    ~DDGIVolume();

    void Initialize(const Settings& settings);
    void Update(const std::vector<class GameObject*>& scene, const std::vector<struct Light>& lights, float deltaTime);
    void MoveTo(const glm::vec3& position); // Update grid center for infinite scrolling
    void Cleanup();
    
    // Bind textures for use in lighting pass
    void BindTextures(int irradianceUnit, int distanceUnit, int probeDataUnit = -1);
    
    // Getters
    const Settings& GetSettings() const { return m_Settings; }
    glm::ivec3 GetGridOrigin() const { return m_GridOrigin; }

private:
    Settings m_Settings;
    bool m_Initialized = false;

    // Toroidal State
    glm::ivec3 m_GridOrigin = glm::ivec3(0); // Integer offset in probe units
    glm::ivec3 m_ResetIndices = glm::ivec3(-1); // Physical indices of planes to reset this frame

    // OpenGL Resources
    unsigned int m_IrradianceTexture = 0; // RGB10_A2 or RGBA16F
    unsigned int m_DistanceTexture = 0;   // RG16F (Mean, Mean^2)
    unsigned int m_ProbeDataSSBO = 0;     // Stores offsets/states (optional)
    unsigned int m_RayHitSSBO = 0;        // Temporary storage for ray results
    unsigned int m_LightSSBO = 0;         // Dynamic lights buffer

    // Shaders
    Shader* m_RaytraceShader = nullptr;
    Shader* m_UpdateShader = nullptr;
    Shader* m_BorderShader = nullptr;

    // Random rotation for rays
    glm::mat4 m_RandomRotation;

    // Helpers
    void CreateTextures();
    void CreateBuffers();
    void LoadShaders();
    void DispatchRaytrace(const std::vector<GameObject*>& scene);
    void DispatchUpdate();
    void DispatchBorderFix();
    void UploadLights(const std::vector<struct Light>& lights);
    
    // Scene management mostly reused from ProbeBaker (uploading scene data to SSBO)
    // We might need a shared GPU scene manager eventually.
    // For now, DDGIVolume will manage its own simple scene upload or reuse existing buffers if refactored.
    // Let's assume we maintain independent buffers for safety for now.
    struct GPUResources {
        unsigned int sceneVertexSSBO;
        unsigned int sceneIndexSSBO;
        unsigned int sceneNormalSSBO;
        unsigned int sceneMaterialSSBO;
        unsigned int bvhSSBO;
        unsigned int primIndexSSBO;
    } m_GPU;
    void UploadScene(const std::vector<GameObject*>& scene);
};
