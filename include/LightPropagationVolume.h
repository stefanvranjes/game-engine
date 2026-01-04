#pragma once

#include <memory>
#include <vector>
#include <glm/glm.hpp>
#include "Shader.h"
#include "Light.h"

/**
 * @class LightPropagationVolume
 * @brief Light Propagation Volumes for fast indirect lighting
 * 
 * Uses Reflective Shadow Maps (RSM) to inject light into a 3D grid,
 * then propagates light using spherical harmonics.
 */
class LightPropagationVolume {
public:
    LightPropagationVolume(int gridSize = 32);
    ~LightPropagationVolume();

    // Initialization
    bool Initialize();
    void Shutdown();

    // Main LPV pipeline
    void Inject(const std::vector<Light>& lights, class Camera* camera);
    void Propagate(int iterations = 4);
    void Clear();

    // Getters
    unsigned int GetLPVTextureR() const { return m_LPVTextureR[m_CurrentBuffer]; }
    unsigned int GetLPVTextureG() const { return m_LPVTextureG[m_CurrentBuffer]; }
    unsigned int GetLPVTextureB() const { return m_LPVTextureB[m_CurrentBuffer]; }
    int GetGridSize() const { return m_GridSize; }
    glm::vec3 GetGridMin() const { return m_GridMin; }
    glm::vec3 GetGridMax() const { return m_GridMax; }

    // Configuration
    void SetGridBounds(const glm::vec3& min, const glm::vec3& max);
    void SetPropagationIterations(int iterations) { m_PropagationIterations = iterations; }

private:
    void CreateLPVTextures();
    void CreateRSMResources();
    void GenerateRSM(const Light& light, Camera* camera);
    void SwapBuffers();

    // Grid properties
    int m_GridSize;
    glm::vec3 m_GridMin;
    glm::vec3 m_GridMax;
    float m_CellSize;

    // LPV 3D textures (ping-pong buffers for propagation)
    // Each stores spherical harmonic coefficients for R, G, B channels
    unsigned int m_LPVTextureR[2];  // Red channel SH coefficients
    unsigned int m_LPVTextureG[2];  // Green channel SH coefficients
    unsigned int m_LPVTextureB[2];  // Blue channel SH coefficients
    int m_CurrentBuffer;

    // Reflective Shadow Map (RSM) resources
    unsigned int m_RSMFramebuffer;
    unsigned int m_RSMPositionTexture;
    unsigned int m_RSMNormalTexture;
    unsigned int m_RSMFluxTexture;
    unsigned int m_RSMDepthTexture;
    int m_RSMResolution;

    // Shaders
    std::unique_ptr<Shader> m_RSMShader;
    std::unique_ptr<Shader> m_InjectShader;
    std::unique_ptr<Shader> m_PropagateShader;

    // Configuration
    int m_PropagationIterations;
};
