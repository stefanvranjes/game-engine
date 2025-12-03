#pragma once

#include "Math/Mat4.h"
#include <memory>

class SSR {
public:
    SSR();
    ~SSR();

    bool Init(unsigned int width, unsigned int height);
    void Render(unsigned int positionTexture, unsigned int normalTexture, 
                unsigned int albedoSpecTexture, const Mat4& view, const Mat4& projection);
    void Shutdown();

    unsigned int GetSSRTexture() const { return m_SSRTexture; }
    
    // Configuration
    void SetMaxSteps(int steps) { m_MaxSteps = steps; }
    void SetStepSize(float size) { m_StepSize = size; }
    void SetThickness(float thickness) { m_Thickness = thickness; }
    void SetMaxDistance(float distance) { m_MaxDistance = distance; }
    void SetFadeStart(float start) { m_FadeStart = start; }
    void SetFadeEnd(float end) { m_FadeEnd = end; }
    
    int GetMaxSteps() const { return m_MaxSteps; }
    float GetStepSize() const { return m_StepSize; }
    float GetThickness() const { return m_Thickness; }
    float GetMaxDistance() const { return m_MaxDistance; }
    float GetFadeStart() const { return m_FadeStart; }
    float GetFadeEnd() const { return m_FadeEnd; }

private:
    unsigned int m_FBO;
    unsigned int m_SSRTexture;
    unsigned int m_BlurTexture;
    unsigned int m_QuadVAO;
    unsigned int m_QuadVBO;
    
    unsigned int m_Width;
    unsigned int m_Height;
    
    std::unique_ptr<class Shader> m_SSRShader;
    std::unique_ptr<class Shader> m_BlurShader;
    
    // Configuration parameters
    int m_MaxSteps;
    float m_StepSize;
    float m_Thickness;
    float m_MaxDistance;
    float m_FadeStart;
    float m_FadeEnd;
    
    void RenderQuad();
};
