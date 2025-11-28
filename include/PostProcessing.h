#pragma once
#include <memory>
#include "GLExtensions.h"

class Shader;

class PostProcessing {
public:
    PostProcessing();
    ~PostProcessing();

    bool Init(int width, int height);
    void Resize(int width, int height);
    
    // Begin rendering to HDR framebuffer
    void BeginHDR();
    
    // Apply all post-processing effects and render to screen
    void ApplyEffects();
    
    // Getters for parameters
    bool IsBloomEnabled() const { return m_BloomEnabled; }
    float GetBloomIntensity() const { return m_BloomIntensity; }
    float GetBloomThreshold() const { return m_BloomThreshold; }
    float GetExposure() const { return m_Exposure; }
    float GetGamma() const { return m_Gamma; }
    int GetToneMappingMode() const { return m_ToneMappingMode; }
    
    // Setters for parameters
    void SetBloomEnabled(bool enabled) { m_BloomEnabled = enabled; }
    void SetBloomIntensity(float intensity) { m_BloomIntensity = intensity; }
    void SetBloomThreshold(float threshold) { m_BloomThreshold = threshold; }
    void SetExposure(float exposure) { m_Exposure = exposure; }
    void SetGamma(float gamma) { m_Gamma = gamma; }
    void SetToneMappingMode(int mode) { m_ToneMappingMode = mode; }

private:
    void CreateFramebuffers();
    void RenderQuad();
    void ExtractBrightness();
    void BlurBrightness();
    void ApplyBloom();
    void ApplyToneMapping();

    int m_Width, m_Height;
    
    // HDR framebuffer
    GLuint m_HDRFBO;
    GLuint m_HDRColorBuffer;
    GLuint m_HDRDepthBuffer;
    
    // Ping-pong framebuffers for blur
    GLuint m_PingPongFBO[2];
    GLuint m_PingPongBuffer[2];
    
    // Brightness extraction framebuffer
    GLuint m_BrightFBO;
    GLuint m_BrightBuffer;
    
    // Quad VAO/VBO for fullscreen rendering
    GLuint m_QuadVAO;
    GLuint m_QuadVBO;
    
    // Shaders
    std::unique_ptr<Shader> m_BrightnessShader;
    std::unique_ptr<Shader> m_BlurShader;
    std::unique_ptr<Shader> m_BloomBlendShader;
    std::unique_ptr<Shader> m_ToneMappingShader;
    
    // Parameters
    bool m_BloomEnabled;
    float m_BloomIntensity;
    float m_BloomThreshold;
    float m_Exposure;
    float m_Gamma;
    int m_ToneMappingMode; // 0 = Reinhard, 1 = ACES
};
