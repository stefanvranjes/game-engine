#include "SSAO.h"
#include "GLExtensions.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <random>

SSAO::SSAO()
    : m_SSAOFBO(0)
    , m_BlurFBO(0)
    , m_SSAOTexture(0)
    , m_BlurTexture(0)
    , m_NoiseTexture(0)
    , m_Width(0)
    , m_Height(0)
    , m_Radius(0.5f)
    , m_Bias(0.025f)
{
}

SSAO::~SSAO() {
    Shutdown();
}

void SSAO::GenerateSampleKernel() {
    std::uniform_real_distribution<float> randomFloats(0.0f, 1.0f);
    std::default_random_engine generator;
    
    m_SampleKernel.clear();
    
    for (unsigned int i = 0; i < 64; ++i) {
        // Generate random sample in hemisphere
        float x = randomFloats(generator) * 2.0f - 1.0f;
        float y = randomFloats(generator) * 2.0f - 1.0f;
        float z = randomFloats(generator);
        
        // Normalize
        float length = std::sqrt(x * x + y * y + z * z);
        x /= length;
        y /= length;
        z /= length;
        
        // Scale samples so they're more aligned to center of kernel
        float scale = (float)i / 64.0f;
        scale = 0.1f + (scale * scale) * 0.9f; // Lerp between 0.1 and 1.0
        
        x *= scale;
        y *= scale;
        z *= scale;
        
        m_SampleKernel.push_back(x);
        m_SampleKernel.push_back(y);
        m_SampleKernel.push_back(z);
    }
}

void SSAO::GenerateNoiseTexture() {
    std::uniform_real_distribution<float> randomFloats(0.0f, 1.0f);
    std::default_random_engine generator;
    
    std::vector<float> ssaoNoise;
    for (unsigned int i = 0; i < 16; i++) {
        // Rotate around z-axis (in tangent space)
        float x = randomFloats(generator) * 2.0f - 1.0f;
        float y = randomFloats(generator) * 2.0f - 1.0f;
        float z = 0.0f;
        
        ssaoNoise.push_back(x);
        ssaoNoise.push_back(y);
        ssaoNoise.push_back(z);
    }
    
    glGenTextures(1, &m_NoiseTexture);
    glBindTexture(GL_TEXTURE_2D, m_NoiseTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 4, 4, 0, GL_RGB, GL_FLOAT, &ssaoNoise[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
}

bool SSAO::Init(unsigned int width, unsigned int height) {
    m_Width = width;
    m_Height = height;
    
    // Generate sample kernel and noise texture
    GenerateSampleKernel();
    GenerateNoiseTexture();
    
    // Create SSAO framebuffer
    glGenFramebuffers(1, &m_SSAOFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_SSAOFBO);
    
    // SSAO color buffer
    glGenTextures(1, &m_SSAOTexture);
    glBindTexture(GL_TEXTURE_2D, m_SSAOTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_SSAOTexture, 0);
    
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "SSAO Framebuffer not complete!" << std::endl;
        return false;
    }
    
    // Create blur framebuffer
    glGenFramebuffers(1, &m_BlurFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_BlurFBO);
    
    glGenTextures(1, &m_BlurTexture);
    glBindTexture(GL_TEXTURE_2D, m_BlurTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_BlurTexture, 0);
    
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "SSAO Blur Framebuffer not complete!" << std::endl;
        return false;
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    // Load shaders
    m_SSAOShader = std::make_unique<Shader>();
    if (!m_SSAOShader->LoadFromFiles("shaders/ssao.vert", "shaders/ssao.frag")) {
        std::cerr << "Failed to load SSAO shaders" << std::endl;
        return false;
    }
    
    m_BlurShader = std::make_unique<Shader>();
    if (!m_BlurShader->LoadFromFiles("shaders/ssao.vert", "shaders/ssao_blur.frag")) {
        std::cerr << "Failed to load SSAO blur shaders" << std::endl;
        return false;
    }
    
    // Set shader uniforms that don't change
    m_SSAOShader->Use();
    m_SSAOShader->SetInt("gPosition", 0);
    m_SSAOShader->SetInt("gNormal", 1);
    m_SSAOShader->SetInt("texNoise", 2);
    
    // Upload sample kernel
    for (unsigned int i = 0; i < 64; ++i) {
        std::string uniformName = "samples[" + std::to_string(i) + "]";
        m_SSAOShader->SetVec3(uniformName, 
            m_SampleKernel[i * 3], 
            m_SampleKernel[i * 3 + 1], 
            m_SampleKernel[i * 3 + 2]);
    }
    
    m_BlurShader->Use();
    m_BlurShader->SetInt("ssaoInput", 0);
    
    return true;
}

void SSAO::Render(unsigned int gPositionTexture, unsigned int gNormalTexture, const float* projection, const float* view) {
    // ===== SSAO Pass =====
    glBindFramebuffer(GL_FRAMEBUFFER, m_SSAOFBO);
    glClear(GL_COLOR_BUFFER_BIT);
    
    m_SSAOShader->Use();
    
    // Bind textures
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gPositionTexture);
    
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, gNormalTexture);
    
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, m_NoiseTexture);
    
    // Set uniforms
    m_SSAOShader->SetMat4("projection", projection);
    m_SSAOShader->SetMat4("view", view);
    m_SSAOShader->SetFloat("radius", m_Radius);
    m_SSAOShader->SetFloat("bias", m_Bias);
    m_SSAOShader->SetVec2("noiseScale", (float)m_Width / 4.0f, (float)m_Height / 4.0f);
    
    // Render fullscreen quad
    glBindVertexArray(0); // Will be handled by Renderer's quad
    // Note: We need to call Renderer's RenderQuad, but since we can't access it,
    // we'll render a simple quad here
    RenderQuad();
    
    // ===== Blur Pass =====
    glBindFramebuffer(GL_FRAMEBUFFER, m_BlurFBO);
    glClear(GL_COLOR_BUFFER_BIT);
    
    m_BlurShader->Use();
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_SSAOTexture);
    
    RenderQuad();
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SSAO::RenderQuad() {
    // Simple quad rendering (we'll use Renderer's quad in integration)
    static unsigned int quadVAO = 0;
    static unsigned int quadVBO = 0;
    
    if (quadVAO == 0) {
        float quadVertices[] = {
            // positions   // texCoords
            -1.0f,  1.0f,  0.0f, 1.0f,
            -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f, 0.0f,

            -1.0f,  1.0f,  0.0f, 1.0f,
             1.0f, -1.0f,  1.0f, 0.0f,
             1.0f,  1.0f,  1.0f, 1.0f
        };
        
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    }
    
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void SSAO::Shutdown() {
    if (m_SSAOFBO) {
        glDeleteFramebuffers(1, &m_SSAOFBO);
        m_SSAOFBO = 0;
    }
    if (m_BlurFBO) {
        glDeleteFramebuffers(1, &m_BlurFBO);
        m_BlurFBO = 0;
    }
    if (m_SSAOTexture) {
        glDeleteTextures(1, &m_SSAOTexture);
        m_SSAOTexture = 0;
    }
    if (m_BlurTexture) {
        glDeleteTextures(1, &m_BlurTexture);
        m_BlurTexture = 0;
    }
    if (m_NoiseTexture) {
        glDeleteTextures(1, &m_NoiseTexture);
        m_NoiseTexture = 0;
    }
}
