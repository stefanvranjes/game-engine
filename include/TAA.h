#pragma once

#include "Math/Mat4.h"
#include "Math/Vec2.h"
#include <memory>

class TAA {
public:
    TAA();
    ~TAA();

    bool Init(unsigned int width, unsigned int height);
    void Render(unsigned int currentFrameTexture, unsigned int velocityTexture,
                const Mat4& view, const Mat4& projection, const Mat4& prevViewProj);
    void Shutdown();

    unsigned int GetOutputTexture() const { return m_HistoryTextures[m_CurrentFrame]; }
    
    // Get jitter for current frame
    Vec2 GetJitter() const;
    
    // Configuration
    void SetBlendFactor(float factor) { m_BlendFactor = factor; }
    void SetJitterScale(float scale) { m_JitterScale = scale; }
    
    float GetBlendFactor() const { return m_BlendFactor; }
    float GetJitterScale() const { return m_JitterScale; }

private:
    unsigned int m_FBO;
    unsigned int m_HistoryTextures[2];  // Double-buffered history
    unsigned int m_QuadVAO;
    unsigned int m_QuadVBO;
    
    unsigned int m_Width;
    unsigned int m_Height;
    
    std::unique_ptr<class Shader> m_TAAShader;
    
    // Frame tracking
    int m_CurrentFrame;
    int m_FrameIndex;
    
    // Configuration
    float m_BlendFactor;    // How much history to keep (0.9 = 90% history)
    float m_JitterScale;    // Scale for jitter pattern
    
    void RenderQuad();
    float Halton(int index, int base) const;
};
