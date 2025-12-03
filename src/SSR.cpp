#include "SSR.h"
#include "Shader.h"
#include "GLExtensions.h"
#include <GLFW/glfw3.h>
#include <iostream>

SSR::SSR()
    : m_FBO(0)
    , m_SSRTexture(0)
    , m_BlurTexture(0)
    , m_QuadVAO(0)
    , m_QuadVBO(0)
    , m_Width(0)
    , m_Height(0)
    , m_MaxSteps(64)
    , m_StepSize(0.1f)
    , m_Thickness(0.5f)
    , m_MaxDistance(50.0f)
    , m_FadeStart(0.7f)
    , m_FadeEnd(1.0f)
{
}

SSR::~SSR() {
    Shutdown();
}

bool SSR::Init(unsigned int width, unsigned int height) {
    m_Width = width;
    m_Height = height;

    // Create framebuffer
    glGenFramebuffers(1, &m_FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);

    // Create SSR texture (RGBA16F for color + alpha for confidence)
    glGenTextures(1, &m_SSRTexture);
    glBindTexture(GL_TEXTURE_2D, m_SSRTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_SSRTexture, 0);

    // Create blur texture
    glGenTextures(1, &m_BlurTexture);
    glBindTexture(GL_TEXTURE_2D, m_BlurTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Check framebuffer completeness
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "SSR framebuffer is not complete!" << std::endl;
        return false;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

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

    // Load shaders
    m_SSRShader = std::make_unique<Shader>();
    if (!m_SSRShader->LoadFromFiles("shaders/ssr.vert", "shaders/ssr.frag")) {
        std::cerr << "Failed to load SSR shaders" << std::endl;
        return false;
    }

    m_BlurShader = std::make_unique<Shader>();
    if (!m_BlurShader->LoadFromFiles("shaders/ssr.vert", "shaders/ssr_blur.frag")) {
        std::cerr << "Failed to load SSR blur shaders" << std::endl;
        return false;
    }

    std::cout << "SSR initialized successfully" << std::endl;
    return true;
}

void SSR::Render(unsigned int positionTexture, unsigned int normalTexture, 
                 unsigned int albedoSpecTexture, const Mat4& view, const Mat4& projection) {
    // ===== Pass 1: SSR Ray Marching =====
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_BlurTexture, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    m_SSRShader->Use();
    m_SSRShader->SetInt("gPosition", 0);
    m_SSRShader->SetInt("gNormal", 1);
    m_SSRShader->SetInt("gAlbedoSpec", 2);
    m_SSRShader->SetMat4("view", view.m);
    m_SSRShader->SetMat4("projection", projection.m);
    m_SSRShader->SetInt("maxSteps", m_MaxSteps);
    m_SSRShader->SetFloat("stepSize", m_StepSize);
    m_SSRShader->SetFloat("thickness", m_Thickness);
    m_SSRShader->SetFloat("maxDistance", m_MaxDistance);
    m_SSRShader->SetFloat("fadeStart", m_FadeStart);
    m_SSRShader->SetFloat("fadeEnd", m_FadeEnd);
    m_SSRShader->SetVec2("screenSize", (float)m_Width, (float)m_Height);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, positionTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, normalTexture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, albedoSpecTexture);

    RenderQuad();

    // ===== Pass 2: Bilateral Blur =====
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_SSRTexture, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    m_BlurShader->Use();
    m_BlurShader->SetInt("ssrTexture", 0);
    m_BlurShader->SetInt("gPosition", 1);
    m_BlurShader->SetVec2("screenSize", (float)m_Width, (float)m_Height);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_BlurTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, positionTexture);

    RenderQuad();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SSR::RenderQuad() {
    glBindVertexArray(m_QuadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void SSR::Shutdown() {
    if (m_FBO) {
        glDeleteFramebuffers(1, &m_FBO);
        m_FBO = 0;
    }
    if (m_SSRTexture) {
        glDeleteTextures(1, &m_SSRTexture);
        m_SSRTexture = 0;
    }
    if (m_BlurTexture) {
        glDeleteTextures(1, &m_BlurTexture);
        m_BlurTexture = 0;
    }
    if (m_QuadVAO) {
        glDeleteVertexArrays(1, &m_QuadVAO);
        m_QuadVAO = 0;
    }
    if (m_QuadVBO) {
        glDeleteBuffers(1, &m_QuadVBO);
        m_QuadVBO = 0;
    }
}
