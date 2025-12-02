#include "GBuffer.h"
#include "GLExtensions.h"
#include <iostream>

GBuffer::GBuffer()
    : m_FBO(0)
    , m_PositionTexture(0)
    , m_NormalTexture(0)
    , m_AlbedoSpecTexture(0)
    , m_EmissiveTexture(0)
    , m_DepthTexture(0)
    , m_Width(0)
    , m_Height(0)
{
}

GBuffer::~GBuffer() {
    Shutdown();
}

bool GBuffer::Init(unsigned int width, unsigned int height) {
    m_Width = width;
    m_Height = height;

    // Create framebuffer
    glGenFramebuffers(1, &m_FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);

    // Position texture (RGBA16F for high precision) - RGB: Position, A: AO
    glGenTextures(1, &m_PositionTexture);
    glBindTexture(GL_TEXTURE_2D, m_PositionTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_PositionTexture, 0);

    // Normal texture (RGBA16F for high precision) - RGB: Normal, A: Roughness
    glGenTextures(1, &m_NormalTexture);
    glBindTexture(GL_TEXTURE_2D, m_NormalTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, m_NormalTexture, 0);

    // Albedo + Metallic texture (RGBA) - RGB: Albedo, A: Metallic
    glGenTextures(1, &m_AlbedoSpecTexture);
    glBindTexture(GL_TEXTURE_2D, m_AlbedoSpecTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, m_AlbedoSpecTexture, 0);

    // Emissive texture (RGB16F for HDR)
    glGenTextures(1, &m_EmissiveTexture);
    glBindTexture(GL_TEXTURE_2D, m_EmissiveTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, m_EmissiveTexture, 0);

    // Depth texture
    glGenTextures(1, &m_DepthTexture);
    glBindTexture(GL_TEXTURE_2D, m_DepthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_DepthTexture, 0);

    // Tell OpenGL which color attachments we'll use
    unsigned int attachments[4] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 };
    glDrawBuffers(4, attachments);

    // Check framebuffer completeness
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "GBuffer framebuffer is not complete!" << std::endl;
        return false;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    std::cout << "GBuffer initialized successfully" << std::endl;
    return true;
}

void GBuffer::BindForWriting() {
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
}

void GBuffer::BindForReading() {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_PositionTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_NormalTexture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, m_AlbedoSpecTexture);
}

void GBuffer::Shutdown() {
    if (m_FBO) {
        glDeleteFramebuffers(1, &m_FBO);
        m_FBO = 0;
    }
    if (m_PositionTexture) {
        glDeleteTextures(1, &m_PositionTexture);
        m_PositionTexture = 0;
    }
    if (m_NormalTexture) {
        glDeleteTextures(1, &m_NormalTexture);
        m_NormalTexture = 0;
    }
    if (m_AlbedoSpecTexture) {
        glDeleteTextures(1, &m_AlbedoSpecTexture);
        m_AlbedoSpecTexture = 0;
    }
    if (m_EmissiveTexture) {
        glDeleteTextures(1, &m_EmissiveTexture);
        m_EmissiveTexture = 0;
    }
    if (m_DepthTexture) {
        glDeleteTextures(1, &m_DepthTexture);
        m_DepthTexture = 0;
    }
}
