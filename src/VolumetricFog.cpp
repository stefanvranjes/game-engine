#include "VolumetricFog.h"
#include "GLExtensions.h"
#include <iostream>
#include <vector>

VolumetricFog::VolumetricFog() 
    : m_FBO(0), m_FogTexture(0), m_Width(0), m_Height(0), 
      m_Density(0.05f), m_Anisotropy(0.8f), m_MaxSteps(100), m_StepSize(0.5f) {
}

VolumetricFog::~VolumetricFog() {
    Shutdown();
}

bool VolumetricFog::Init(unsigned int width, unsigned int height) {
    // Determine fog resolution (half resolution is common for volumetric)
    m_Width = width / 2;
    m_Height = height / 2;
    
    // Create Shader
    m_Shader = std::make_unique<Shader>();
    if (!m_Shader->LoadFromFiles("shaders/volumetric_fog.vert", "shaders/volumetric_fog.frag")) {
        std::cerr << "Failed to load volumetric fog shaders!" << std::endl;
        return false;
    }
    
    glGenFramebuffers(1, &m_FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    
    glGenTextures(1, &m_FogTexture);
    glBindTexture(GL_TEXTURE_2D, m_FogTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, m_Width, m_Height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_FogTexture, 0);
    
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Volumetric Fog Framebuffer not complete!" << std::endl;
        return false;
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    return true;
}

void VolumetricFog::Resize(unsigned int width, unsigned int height) {
    m_Width = width / 2;
    m_Height = height / 2;
    
    glBindTexture(GL_TEXTURE_2D, m_FogTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, m_Width, m_Height, 0, GL_RGBA, GL_FLOAT, NULL);
}

void VolumetricFog::Render(unsigned int depthTexture, unsigned int shadowMapTexture, 
            const float* view, const float* projection, const float* inverseView, const float* inverseProjection,
            const float* viewPos, 
            const float* lightDir, const float* lightColor, float lightIntensity,
            const float* cascadeMatrices, const float* cascadeDistances) {
                
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glClear(GL_COLOR_BUFFER_BIT); // Accumulation
    glViewport(0, 0, m_Width, m_Height);
    
    m_Shader->Use();
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    m_Shader->SetInt("gDepth", 0);
    
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D_ARRAY, shadowMapTexture);
    m_Shader->SetInt("shadowMap", 1);
    
    m_Shader->SetMat4("inverseView", inverseView);
    m_Shader->SetMat4("inverseProjection", inverseProjection);
    m_Shader->SetVec3("viewPos", viewPos[0], viewPos[1], viewPos[2]);
    
    m_Shader->SetVec3("lightDir", lightDir[0], lightDir[1], lightDir[2]);
    m_Shader->SetVec3("lightColor", lightColor[0], lightColor[1], lightColor[2]);
    m_Shader->SetFloat("lightIntensity", lightIntensity);
    
    // Set Shadow uniforms
    // uniform mat4 cascadeLightSpaceMatrices[3];
    // uniform float cascadePlaneDistances[3];
    // The shader expects arrays. We can pass them as arrays using glUniformMatrix4fv etc.
    // Shader class abstraction likely handles SetMat4 for single, need to check if it handles arrays.
    // Standard Shader::SetMat4 usually is for one. I might need a SetMat4Array or loop.
    // Assuming Render.cpp does: m_LightingShader->SetMat4("cascadeLightSpaceMatrices[" + std::to_string(i) + "]", ...)
    // I will use that pattern.
    
    // cast back to Mat4 structure or just unsafe pointer arithmetic
    for (int i = 0; i < 3; ++i) {
        // Each Mat4 is 16 floats
        m_Shader->SetMat4("cascadeLightSpaceMatrices[" + std::to_string(i) + "]", &cascadeMatrices[i * 16]);
        m_Shader->SetFloat("cascadePlaneDistances[" + std::to_string(i) + "]", cascadeDistances[i]);
    }
    
    m_Shader->SetFloat("u_Density", m_Density);
    m_Shader->SetFloat("u_Anisotropy", m_Anisotropy);
    m_Shader->SetInt("u_MaxSteps", m_MaxSteps);
    m_Shader->SetFloat("u_StepSize", m_StepSize);
    
    RenderQuad();
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void VolumetricFog::RenderQuad() {
    if (m_QuadVAO == 0) {
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
    }
    glBindVertexArray(m_QuadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void VolumetricFog::Shutdown() {
    if (m_FBO) glDeleteFramebuffers(1, &m_FBO);
    if (m_FogTexture) glDeleteTextures(1, &m_FogTexture);
    if (m_QuadVAO) glDeleteVertexArrays(1, &m_QuadVAO);
    if (m_QuadVBO) glDeleteBuffers(1, &m_QuadVBO);
}
