#include "CascadedShadowMap.h"
#include "GLExtensions.h"
#include <iostream>

CascadedShadowMap::CascadedShadowMap() : m_FBO(0), m_DepthMapArray(0), m_Width(0), m_Height(0) {
}

CascadedShadowMap::~CascadedShadowMap() {
    if (m_FBO) glDeleteFramebuffers(1, &m_FBO);
    if (m_DepthMapArray) glDeleteTextures(1, &m_DepthMapArray);
}

bool CascadedShadowMap::Init(unsigned int width, unsigned int height) {
    m_Width = width;
    m_Height = height;

    // Create framebuffer
    glGenFramebuffers(1, &m_FBO);

    // Create depth texture array
    glGenTextures(1, &m_DepthMapArray);
    glBindTexture(GL_TEXTURE_2D_ARRAY, m_DepthMapArray);
    
    // Allocate storage for 3 cascades
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT32F, width, height, 3, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, borderColor);

    // Attach to framebuffer (we'll attach specific layers when rendering)
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, m_DepthMapArray, 0, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Cascaded Shadow Map framebuffer is not complete!" << std::endl;
        return false;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return true;
}

void CascadedShadowMap::BindForWriting(unsigned int cascadeIndex) {
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, m_DepthMapArray, 0, cascadeIndex);
}

void CascadedShadowMap::BindForReading(unsigned int textureUnit) {
    glActiveTexture(GL_TEXTURE0 + textureUnit);
    glBindTexture(GL_TEXTURE_2D_ARRAY, m_DepthMapArray);
}
