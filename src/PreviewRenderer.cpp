#include "PreviewRenderer.h"
#include "GLExtensions.h"
#include <GLFW/glfw3.h>
#include <iostream>

PreviewRenderer::PreviewRenderer()
    : m_FBO(0)
    , m_ColorTexture(0)
    , m_DepthRenderbuffer(0)
    , m_Width(512)
    , m_Height(512)
    , m_RotationAngle(0.0f)
{
}

PreviewRenderer::~PreviewRenderer() {
    Shutdown();
}

bool PreviewRenderer::Init(int width, int height) {
    m_Width = width;
    m_Height = height;

    // Create framebuffer
    glGenFramebuffers(1, &m_FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);

    // Create color texture
    glGenTextures(1, &m_ColorTexture);
    glBindTexture(GL_TEXTURE_2D, m_ColorTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_Width, m_Height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_ColorTexture, 0);

    // Create depth renderbuffer
    glGenRenderbuffers(1, &m_DepthRenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, m_DepthRenderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, m_Width, m_Height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_DepthRenderbuffer);

    // Check framebuffer completeness
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Preview framebuffer is not complete!" << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return false;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Create preview camera
    m_Camera = std::make_unique<Camera>(Vec3(0, 0, 3), 45.0f, (float)m_Width / (float)m_Height);

    // Create simple shader for preview
    m_Shader = std::make_unique<Shader>();
    if (!m_Shader->LoadFromFiles("shaders/textured.vert", "shaders/textured.frag")) {
        std::cerr << "Failed to load preview shader" << std::endl;
        return false;
    }

    std::cout << "PreviewRenderer initialized (" << m_Width << "x" << m_Height << ")" << std::endl;
    return true;
}

void PreviewRenderer::RenderPreview(GameObject* object, Shader* shader) {
    if (!object || m_FBO == 0 || !m_Shader) return;

    // Use internal shader
    Shader* renderShader = m_Shader.get();

    // Update rotation
    m_RotationAngle += 0.5f;
    if (m_RotationAngle > 360.0f) m_RotationAngle -= 360.0f;

    // Position camera in orbit
    float radius = 3.0f;
    float angleRad = m_RotationAngle * 3.14159f / 180.0f;
    Vec3 camPos(
        radius * sin(angleRad),
        1.0f,
        radius * cos(angleRad)
    );
    m_Camera->SetPosition(camPos);
    // Camera automatically looks at origin based on yaw/pitch

    // Bind preview framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glViewport(0, 0, m_Width, m_Height);

    // Clear
    glClearColor(0.2f, 0.2f, 0.25f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Enable depth test
    glEnable(GL_DEPTH_TEST);

    // Use shader
    renderShader->Use();

    // Set matrices
    Mat4 view = m_Camera->GetViewMatrix();
    Mat4 projection = m_Camera->GetProjectionMatrix();
    Mat4 model; // Identity - object at origin

    renderShader->SetMat4("u_View", view.m);
    renderShader->SetMat4("u_Projection", projection.m);
    renderShader->SetMat4("u_Model", model.m);

    // Simple directional light
    renderShader->SetVec3("u_LightDir", 0.3f, -1.0f, 0.5f);
    renderShader->SetVec3("u_LightColor", 1.0f, 1.0f, 1.0f);
    renderShader->SetVec3("u_ViewPos", camPos.x, camPos.y, camPos.z);

    // Bind material and draw
    auto material = object->GetMaterial();
    if (material) {
        material->Bind(renderShader);
    }

    auto mesh = object->GetActiveMesh(view);
    if (mesh) {
        mesh->Draw();
    }

    // Unbind framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void PreviewRenderer::Shutdown() {
    if (m_FBO != 0) {
        glDeleteFramebuffers(1, &m_FBO);
        m_FBO = 0;
    }
    if (m_ColorTexture != 0) {
        glDeleteTextures(1, &m_ColorTexture);
        m_ColorTexture = 0;
    }
    if (m_DepthRenderbuffer != 0) {
        glDeleteRenderbuffers(1, &m_DepthRenderbuffer);
        m_DepthRenderbuffer = 0;
    }
}
