#include "PostProcessing.h"
#include "Shader.h"
#include <iostream>

PostProcessing::PostProcessing()
    : m_Width(0), m_Height(0)
    , m_HDRFBO(0), m_HDRColorBuffer(0), m_HDRDepthBuffer(0)
    , m_BrightFBO(0), m_BrightBuffer(0)
    , m_QuadVAO(0), m_QuadVBO(0)
    , m_BloomEnabled(true)
    , m_BloomIntensity(0.5f)
    , m_BloomThreshold(1.0f)
    , m_Exposure(1.0f)
    , m_Gamma(2.2f)
    , m_ToneMappingMode(0)
{
    m_PingPongFBO[0] = m_PingPongFBO[1] = 0;
    m_PingPongBuffer[0] = m_PingPongBuffer[1] = 0;
}

PostProcessing::~PostProcessing() {
    if (m_HDRFBO) glDeleteFramebuffers(1, &m_HDRFBO);
    if (m_HDRColorBuffer) glDeleteTextures(1, &m_HDRColorBuffer);
    if (m_HDRDepthBuffer) glDeleteRenderbuffers(1, &m_HDRDepthBuffer);
    if (m_BrightFBO) glDeleteFramebuffers(1, &m_BrightFBO);
    if (m_BrightBuffer) glDeleteTextures(1, &m_BrightBuffer);
    if (m_PingPongFBO[0]) glDeleteFramebuffers(2, m_PingPongFBO);
    if (m_PingPongBuffer[0]) glDeleteTextures(2, m_PingPongBuffer);
    if (m_QuadVAO) glDeleteVertexArrays(1, &m_QuadVAO);
    if (m_QuadVBO) glDeleteBuffers(1, &m_QuadVBO);
}

bool PostProcessing::Init(int width, int height) {
    m_Width = width;
    m_Height = height;
    
    // Load shaders
    m_BrightnessShader = std::make_unique<Shader>();
    if (!m_BrightnessShader->LoadFromFiles("shaders/hdr.vert", "shaders/bloom_extract.frag")) {
        std::cerr << "Failed to load brightness extraction shader" << std::endl;
        return false;
    }
    
    m_BlurShader = std::make_unique<Shader>();
    if (!m_BlurShader->LoadFromFiles("shaders/hdr.vert", "shaders/blur.frag")) {
        std::cerr << "Failed to load blur shader" << std::endl;
        return false;
    }
    
    m_BloomBlendShader = std::make_unique<Shader>();
    if (!m_BloomBlendShader->LoadFromFiles("shaders/hdr.vert", "shaders/bloom_blend.frag")) {
        std::cerr << "Failed to load bloom blend shader" << std::endl;
        return false;
    }
    
    m_ToneMappingShader = std::make_unique<Shader>();
    if (!m_ToneMappingShader->LoadFromFiles("shaders/hdr.vert", "shaders/hdr.frag")) {
        std::cerr << "Failed to load tone mapping shader" << std::endl;
        return false;
    }
    
    m_UnderwaterShader = std::make_unique<Shader>();
    if (!m_UnderwaterShader->LoadFromFiles("shaders/underwater.vert", "shaders/underwater.frag")) {
        std::cerr << "Failed to load underwater shader" << std::endl;
        // Non-fatal, underwater effects will just be disabled
    }
    
    // Create framebuffers
    CreateFramebuffers();
    
    // Setup fullscreen quad
    float quadVertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    
    glGenVertexArrays(1, &m_QuadVAO);
    glGenBuffers(1, &m_QuadVBO);
    glBindVertexArray(m_QuadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_QuadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glBindVertexArray(0);
    
    std::cout << "Post-processing initialized" << std::endl;
    return true;
}

void PostProcessing::CreateFramebuffers() {
    // HDR framebuffer
    glGenFramebuffers(1, &m_HDRFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_HDRFBO);
    
    // Create floating point color buffer
    glGenTextures(1, &m_HDRColorBuffer);
    glBindTexture(GL_TEXTURE_2D, m_HDRColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, m_Width, m_Height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_HDRColorBuffer, 0);
    
    // Create depth buffer
    glGenRenderbuffers(1, &m_HDRDepthBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, m_HDRDepthBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, m_Width, m_Height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_HDRDepthBuffer);
    
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "HDR Framebuffer not complete!" << std::endl;
    
    // Brightness extraction framebuffer
    glGenFramebuffers(1, &m_BrightFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_BrightFBO);
    
    glGenTextures(1, &m_BrightBuffer);
    glBindTexture(GL_TEXTURE_2D, m_BrightBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, m_Width, m_Height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_BrightBuffer, 0);
    
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "Brightness Framebuffer not complete!" << std::endl;
    
    // Ping-pong framebuffers for blur
    glGenFramebuffers(2, m_PingPongFBO);
    glGenTextures(2, m_PingPongBuffer);
    for (int i = 0; i < 2; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, m_PingPongFBO[i]);
        glBindTexture(GL_TEXTURE_2D, m_PingPongBuffer[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, m_Width, m_Height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_PingPongBuffer[i], 0);
        
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cerr << "Ping-pong Framebuffer " << i << " not complete!" << std::endl;
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void PostProcessing::Resize(int width, int height) {
    if (width == m_Width && height == m_Height) return;
    
    m_Width = width;
    m_Height = height;
    
    // Delete old framebuffers
    if (m_HDRColorBuffer) glDeleteTextures(1, &m_HDRColorBuffer);
    if (m_HDRDepthBuffer) glDeleteRenderbuffers(1, &m_HDRDepthBuffer);
    if (m_BrightBuffer) glDeleteTextures(1, &m_BrightBuffer);
    if (m_PingPongBuffer[0]) glDeleteTextures(2, m_PingPongBuffer);
    
    // Recreate with new size
    CreateFramebuffers();
}

void PostProcessing::BeginHDR() {
    glBindFramebuffer(GL_FRAMEBUFFER, m_HDRFBO);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void PostProcessing::ApplyEffects() {
    glDisable(GL_DEPTH_TEST);
    
    if (m_BloomEnabled) {
        // Step 1: Extract bright areas
        ExtractBrightness();
        
        // Step 2: Blur bright areas
        BlurBrightness();
        
        // Step 3: Blend bloom with HDR image
        ApplyBloom();
    } else {
        // Just apply tone mapping without bloom
        ApplyToneMapping();
    }
    
    glEnable(GL_DEPTH_TEST);
}

void PostProcessing::ExtractBrightness() {
    glBindFramebuffer(GL_FRAMEBUFFER, m_BrightFBO);
    glClear(GL_COLOR_BUFFER_BIT);
    
    m_BrightnessShader->Use();
    m_BrightnessShader->SetInt("hdrBuffer", 0);
    m_BrightnessShader->SetFloat("threshold", m_BloomThreshold);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_HDRColorBuffer);
    
    RenderQuad();
}

void PostProcessing::BlurBrightness() {
    bool horizontal = true;
    int amount = 10; // Number of blur passes
    
    m_BlurShader->Use();
    for (int i = 0; i < amount; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, m_PingPongFBO[horizontal]);
        m_BlurShader->SetInt("horizontal", horizontal);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, i == 0 ? m_BrightBuffer : m_PingPongBuffer[!horizontal]);
        
        RenderQuad();
        horizontal = !horizontal;
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void PostProcessing::ApplyBloom() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    
    m_BloomBlendShader->Use();
    m_BloomBlendShader->SetInt("hdrBuffer", 0);
    m_BloomBlendShader->SetInt("bloomBlur", 1);
    m_BloomBlendShader->SetFloat("bloomIntensity", m_BloomIntensity);
    m_BloomBlendShader->SetFloat("exposure", m_Exposure);
    m_BloomBlendShader->SetFloat("gamma", m_Gamma);
    m_BloomBlendShader->SetInt("toneMappingMode", m_ToneMappingMode);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_HDRColorBuffer);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_PingPongBuffer[0]);
    
    RenderQuad();
}

void PostProcessing::ApplyToneMapping() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    
    m_ToneMappingShader->Use();
    m_ToneMappingShader->SetInt("hdrBuffer", 0);
    m_ToneMappingShader->SetFloat("exposure", m_Exposure);
    m_ToneMappingShader->SetFloat("gamma", m_Gamma);
    m_ToneMappingShader->SetInt("toneMappingMode", m_ToneMappingMode);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_HDRColorBuffer);
    
    RenderQuad();
}

void PostProcessing::RenderQuad() {
    glBindVertexArray(m_QuadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void PostProcessing::ApplyUnderwaterEffect(const UnderwaterParams& params, GLuint depthTexture) {
    if (!params.isUnderwater || !m_UnderwaterShader) {
        return;
    }
    
    glDisable(GL_DEPTH_TEST);
    
    // We apply underwater effect to the HDR buffer in-place
    // by rendering to a ping-pong buffer then copying back
    glBindFramebuffer(GL_FRAMEBUFFER, m_PingPongFBO[0]);
    glClear(GL_COLOR_BUFFER_BIT);
    
    m_UnderwaterShader->Use();
    
    // Bind scene texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_HDRColorBuffer);
    m_UnderwaterShader->SetInt("u_SceneTexture", 0);
    
    // Bind depth texture
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    m_UnderwaterShader->SetInt("u_DepthTexture", 1);
    
    // Set uniforms
    m_UnderwaterShader->SetInt("u_IsUnderwater", 1);
    m_UnderwaterShader->SetFloat("u_Time", params.time);
    m_UnderwaterShader->SetFloat("u_NearPlane", params.nearPlane);
    m_UnderwaterShader->SetFloat("u_FarPlane", params.farPlane);
    m_UnderwaterShader->SetVec3("u_UnderwaterTint", params.tintR, params.tintG, params.tintB);
    m_UnderwaterShader->SetFloat("u_FogDensity", params.fogDensity);
    m_UnderwaterShader->SetFloat("u_FogStart", params.fogStart);
    m_UnderwaterShader->SetFloat("u_FogEnd", params.fogEnd);
    m_UnderwaterShader->SetFloat("u_Distortion", params.distortion);
    m_UnderwaterShader->SetFloat("u_DistortionSpeed", params.distortionSpeed);
    m_UnderwaterShader->SetFloat("u_Vignette", params.vignette);
    
    RenderQuad();
    
    // Copy result back to HDR buffer
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_PingPongFBO[0]);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_HDRFBO);
    glBlitFramebuffer(0, 0, m_Width, m_Height, 0, 0, m_Width, m_Height, GL_COLOR_BUFFER_BIT, GL_LINEAR);
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glEnable(GL_DEPTH_TEST);
}
