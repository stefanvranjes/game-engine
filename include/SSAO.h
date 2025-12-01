#pragma once

#include "Shader.h"
#include <memory>
#include <vector>

class SSAO {
public:
    SSAO();
    ~SSAO();

    bool Init(unsigned int width, unsigned int height);
    void Render(unsigned int gPositionTexture, unsigned int gNormalTexture, const float* projection, const float* view);
    void Shutdown();

    unsigned int GetSSAOTexture() const { return m_BlurTexture; }
    
    // Parameter controls
    void SetRadius(float radius) { m_Radius = radius; }
    void SetBias(float bias) { m_Bias = bias; }
    float GetRadius() const { return m_Radius; }
    float GetBias() const { return m_Bias; }

private:
    void GenerateSampleKernel();
    void GenerateNoiseTexture();
    void RenderQuad();

    unsigned int m_SSAOFBO;
    unsigned int m_BlurFBO;
    unsigned int m_SSAOTexture;
    unsigned int m_BlurTexture;
    unsigned int m_NoiseTexture;
    
    unsigned int m_Width;
    unsigned int m_Height;
    
    std::unique_ptr<Shader> m_SSAOShader;
    std::unique_ptr<Shader> m_BlurShader;
    
    std::vector<float> m_SampleKernel; // 64 samples * 3 components
    
    float m_Radius;
    float m_Bias;
};
