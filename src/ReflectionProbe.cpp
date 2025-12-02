#include "ReflectionProbe.h"
#include "Renderer.h"
#include "GLExtensions.h"
#include "Math/Mat4.h"
#include <iostream>

ReflectionProbe::ReflectionProbe(const Vec3& position, float radius, unsigned int resolution)
    : m_Position(position)
    , m_Radius(radius)
    , m_Resolution(resolution)
    , m_Cubemap(0)
    , m_FBO(0)
    , m_RBO(0)
    , m_NeedsUpdate(true)
{
}

ReflectionProbe::~ReflectionProbe() {
    if (m_Cubemap) glDeleteTextures(1, &m_Cubemap);
    if (m_FBO) glDeleteFramebuffers(1, &m_FBO);
    if (m_RBO) glDeleteRenderbuffers(1, &m_RBO);
}

bool ReflectionProbe::Init() {
    // Create cubemap texture for reflections
    glGenTextures(1, &m_Cubemap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_Cubemap);
    
    for (unsigned int i = 0; i < 6; ++i) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 
                     m_Resolution, m_Resolution, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // Create FBO for rendering
    glGenFramebuffers(1, &m_FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    
    // Create RBO for depth
    glGenRenderbuffers(1, &m_RBO);
    glBindRenderbuffer(GL_RENDERBUFFER, m_RBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, m_Resolution, m_Resolution);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_RBO);
    
    // Check framebuffer completeness
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "ReflectionProbe framebuffer is not complete!" << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return false;
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    std::cout << "ReflectionProbe initialized at position (" 
              << m_Position.x << ", " << m_Position.y << ", " << m_Position.z 
              << ") with radius " << m_Radius << std::endl;
    
    return true;
}

void ReflectionProbe::Capture(Renderer* renderer) {
    // Capture implementation will be in Renderer::CaptureProbe
    // This is just a placeholder that marks the probe as updated
    m_NeedsUpdate = false;
}
