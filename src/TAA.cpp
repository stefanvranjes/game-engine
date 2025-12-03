#include "TAA.h"
#include "Shader.h"
#include "GLExtensions.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>

TAA::TAA()
    : m_FBO(0)
    , m_QuadVAO(0)
    , m_QuadVBO(0)
    , m_Width(0)
    , m_Height(0)
    , m_CurrentFrame(0)
    , m_FrameIndex(0)
    , m_BlendFactor(0.9f)
    , m_JitterScale(1.0f)
{
    m_HistoryTextures[0] = 0;
    m_HistoryTextures[1] = 0;
}

TAA::~TAA() {
    Shutdown();
}

bool TAA::Init(unsigned int width, unsigned int height) {
    m_Width = width;
    m_Height = height;

    // Create framebuffer
    glGenFramebuffers(1, &m_FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);

    // Create history textures (double-buffered)
    for (int i = 0; i < 2; i++) {
        glGenTextures(1, &m_HistoryTextures[i]);
        glBindTexture(GL_TEXTURE_2D, m_HistoryTextures[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    // Attach first texture to check framebuffer completeness
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_HistoryTextures[0], 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "TAA framebuffer is not complete!" << std::endl;
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

    // Load shader
    m_TAAShader = std::make_unique<Shader>();
    if (!m_TAAShader->LoadFromFiles("shaders/taa.vert", "shaders/taa.frag")) {
        std::cerr << "Failed to load TAA shaders" << std::endl;
        return false;
    }

    std::cout << "TAA initialized successfully" << std::endl;
    return true;
}

void TAA::Render(unsigned int currentFrameTexture, unsigned int velocityTexture,
                 const Mat4& view, const Mat4& projection, const Mat4& prevViewProj) {
    // Bind framebuffer with current output texture
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 
                          m_HistoryTextures[m_CurrentFrame], 0);
    glClear(GL_COLOR_BUFFER_BIT);

    m_TAAShader->Use();
    m_TAAShader->SetInt("currentFrame", 0);
    m_TAAShader->SetInt("historyFrame", 1);
    m_TAAShader->SetInt("velocityTexture", 2);
    m_TAAShader->SetFloat("blendFactor", m_BlendFactor);
    m_TAAShader->SetVec2("screenSize", (float)m_Width, (float)m_Height);
    m_TAAShader->SetInt("frameIndex", m_FrameIndex);

    // Bind textures
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, currentFrameTexture);
    
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_HistoryTextures[1 - m_CurrentFrame]); // Previous frame
    
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, velocityTexture);

    RenderQuad();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Swap buffers for next frame
    m_CurrentFrame = 1 - m_CurrentFrame;
    m_FrameIndex++;
}

Vec2 TAA::GetJitter() const {
    // Halton sequence for optimal sample distribution
    // Use base 2 and 3 for x and y
    float x = Halton(m_FrameIndex % 16, 2);
    float y = Halton(m_FrameIndex % 16, 3);
    
    // Convert from [0,1] to [-0.5, 0.5] and scale
    x = (x - 0.5f) * m_JitterScale;
    y = (y - 0.5f) * m_JitterScale;
    
    // Scale to pixel size
    x /= (float)m_Width;
    y /= (float)m_Height;
    
    return Vec2(x, y);
}

float TAA::Halton(int index, int base) const {
    float result = 0.0f;
    float f = 1.0f / (float)base;
    int i = index + 1; // Start from 1 to avoid 0
    
    while (i > 0) {
        result += f * (float)(i % base);
        i = i / base;
        f = f / (float)base;
    }
    
    return result;
}

void TAA::RenderQuad() {
    glBindVertexArray(m_QuadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void TAA::Shutdown() {
    if (m_FBO) {
        glDeleteFramebuffers(1, &m_FBO);
        m_FBO = 0;
    }
    for (int i = 0; i < 2; i++) {
        if (m_HistoryTextures[i]) {
            glDeleteTextures(1, &m_HistoryTextures[i]);
            m_HistoryTextures[i] = 0;
        }
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
