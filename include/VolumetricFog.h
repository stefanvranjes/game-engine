#pragma once

#include "Shader.h"
#include <memory>
#include <vector>

class VolumetricFog {
public:
    VolumetricFog();
    ~VolumetricFog();

    bool Init(unsigned int width, unsigned int height);
    // Assumes cascading shadow maps
    void Render(unsigned int depthTexture, unsigned int shadowMapTexture, 
                const float* view, const float* projection, const float* inverseView, const float* inverseProjection,
                const float* viewPos, 
                const float* lightDir, const float* lightColor, float lightIntensity,
                const float* cascadeMatrices, const float* cascadeDistances);
    
    void Resize(unsigned int width, unsigned int height);
    void Shutdown();

    unsigned int GetFogTexture() const { return m_FogTexture; }

    // Parameters
    void SetDensity(float density) { m_Density = density; }
    void SetAnisotropy(float anisotropy) { m_Anisotropy = anisotropy; }
    void SetMaxSteps(int steps) { m_MaxSteps = steps; }
    void SetStepSize(float stepSize) { m_StepSize = stepSize; }

    float GetDensity() const { return m_Density; }
    float GetAnisotropy() const { return m_Anisotropy; }
    int GetMaxSteps() const { return m_MaxSteps; }
    float GetStepSize() const { return m_StepSize; }

    // Height Fog Parameters
    void SetFogHeight(float height) { m_FogHeight = height; }
    void SetHeightFalloff(float falloff) { m_HeightFalloff = falloff; }
    void SetHeightFogDensity(float density) { m_HeightFogDensity = density; }

    float GetFogHeight() const { return m_FogHeight; }
    float GetHeightFalloff() const { return m_HeightFalloff; }
    float GetHeightFogDensity() const { return m_HeightFogDensity; }

private:
    void RenderQuad();

    unsigned int m_FBO;
    unsigned int m_FogTexture;
    
    unsigned int m_Width;
    unsigned int m_Height;
    
    std::unique_ptr<Shader> m_Shader;
    
    // Config
    // Config
    float m_Density; // Base global density (Exponential factor 1)
    float m_Anisotropy;
    int m_MaxSteps;
    float m_StepSize;

    // Height Fog Config
    float m_FogHeight;         // Base height (Y)
    float m_HeightFalloff;  // How quickly it thins out as you go up
    float m_HeightFogDensity; // Multiplier for the height fog component
    
    // Quad resources
    unsigned int m_QuadVAO = 0;
    unsigned int m_QuadVBO = 0;
};
