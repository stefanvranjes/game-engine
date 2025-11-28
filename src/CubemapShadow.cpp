#include "CubemapShadow.h"
#include <iostream>
#include <vector>

CubemapShadow::CubemapShadow()
    : m_Width(0), m_Height(0), m_FBO(0), m_DepthCubemap(0), m_FarPlane(25.0f)
{
}

CubemapShadow::~CubemapShadow() {
    if (m_FBO) glDeleteFramebuffers(1, &m_FBO);
    if (m_DepthCubemap) glDeleteTextures(1, &m_DepthCubemap);
}

bool CubemapShadow::Init(unsigned int width, unsigned int height) {
    m_Width = width;
    m_Height = height;

    glGenFramebuffers(1, &m_FBO);
    
    glGenTextures(1, &m_DepthCubemap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_DepthCubemap);
    
    for (unsigned int i = 0; i < 6; ++i) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_DEPTH_COMPONENT, 
                     width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    }
    
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, m_DepthCubemap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Cubemap Shadow Framebuffer not complete!" << std::endl;
        return false;
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return true;
}

void CubemapShadow::BindForWriting() {
    glViewport(0, 0, m_Width, m_Height);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glClear(GL_DEPTH_BUFFER_BIT);
}

void CubemapShadow::BindForReading(unsigned int textureUnit) {
    glActiveTexture(GL_TEXTURE0 + textureUnit);
    glBindTexture(GL_TEXTURE_CUBE_MAP, m_DepthCubemap);
}

void CubemapShadow::CalculateViewMatrices(const Vec3& lightPos, std::vector<Mat4>& outTransforms, float& outFarPlane) {
    outFarPlane = m_FarPlane;
    float aspect = (float)m_Width / (float)m_Height;
    float nearPlane = 1.0f;
    Mat4 shadowProj = Mat4::Perspective(90.0f, aspect, nearPlane, m_FarPlane);
    
    outTransforms.clear();
    outTransforms.push_back(shadowProj * Mat4::LookAt(lightPos, lightPos + Vec3( 1.0f,  0.0f,  0.0f), Vec3(0.0f, -1.0f,  0.0f)));
    outTransforms.push_back(shadowProj * Mat4::LookAt(lightPos, lightPos + Vec3(-1.0f,  0.0f,  0.0f), Vec3(0.0f, -1.0f,  0.0f)));
    outTransforms.push_back(shadowProj * Mat4::LookAt(lightPos, lightPos + Vec3( 0.0f,  1.0f,  0.0f), Vec3(0.0f,  0.0f,  1.0f)));
    outTransforms.push_back(shadowProj * Mat4::LookAt(lightPos, lightPos + Vec3( 0.0f, -1.0f,  0.0f), Vec3(0.0f,  0.0f, -1.0f)));
    outTransforms.push_back(shadowProj * Mat4::LookAt(lightPos, lightPos + Vec3( 0.0f,  0.0f,  1.0f), Vec3(0.0f, -1.0f,  0.0f)));
    outTransforms.push_back(shadowProj * Mat4::LookAt(lightPos, lightPos + Vec3( 0.0f,  0.0f, -1.0f), Vec3(0.0f, -1.0f,  0.0f)));
}
