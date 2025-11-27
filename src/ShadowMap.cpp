#include "ShadowMap.h"
#include "GLExtensions.h"
#include <iostream>

ShadowMap::ShadowMap() : m_FBO(0), m_DepthMap(0), m_Width(0), m_Height(0) {
}

ShadowMap::~ShadowMap() {
    if (m_FBO) glDeleteFramebuffers(1, &m_FBO);
    if (m_DepthMap) glDeleteTextures(1, &m_DepthMap);
}

bool ShadowMap::Init(unsigned int width, unsigned int height) {
    m_Width = width;
    m_Height = height;

    // Create framebuffer
    glGenFramebuffers(1, &m_FBO);

    // Create depth texture
    glGenTextures(1, &m_DepthMap);
    glBindTexture(GL_TEXTURE_2D, m_DepthMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Attach depth texture to framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_DepthMap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

    // Check framebuffer completeness
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Shadow map framebuffer is not complete!" << std::endl;
        return false;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return true;
}

void ShadowMap::BindForWriting() {
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
}

void ShadowMap::BindForReading(unsigned int textureUnit) {
    glActiveTexture(GL_TEXTURE0 + textureUnit);
    glBindTexture(GL_TEXTURE_2D, m_DepthMap);
}
