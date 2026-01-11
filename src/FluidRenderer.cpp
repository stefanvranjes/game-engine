#include "FluidRenderer.h"
#include "FluidSimulation.h"
#include "GLExtensions.h"
#include <GL/gl.h>

FluidRenderer::FluidRenderer()
    : m_RenderMode(RenderMode::Particles)
    , m_ScreenWidth(1920)
    , m_ScreenHeight(1080)
    , m_ParticleSize(0.05f)
    , m_SmoothingRadius(2.0f)
    , m_ThicknessScale(1.0f)
    , m_RefractiveIndex(1.33f)
    , m_AbsorptionColor(0.4f, 0.8f, 1.0f)
    , m_FresnelPower(5.0f)
    , m_DepthFBO(0)
    , m_DepthTexture(0)
    , m_SmoothedDepthFBO(0)
    , m_SmoothedDepthTexture(0)
    , m_NormalFBO(0)
    , m_NormalTexture(0)
    , m_ThicknessFBO(0)
    , m_ThicknessTexture(0)
    , m_ParticleVAO(0)
    , m_ParticleVBO(0)
    , m_ParticleInstanceVBO(0)
    , m_QuadVAO(0)
    , m_QuadVBO(0)
{
}

FluidRenderer::~FluidRenderer() {
}

void FluidRenderer::Initialize(int screenWidth, int screenHeight) {
    m_ScreenWidth = screenWidth;
    m_ScreenHeight = screenHeight;
    
    // Load shaders
    m_ParticleShader = std::make_unique<Shader>("shaders/particle.vert", "shaders/particle.frag");
    m_DepthShader = std::make_unique<Shader>("shaders/fluid_depth.vert", "shaders/fluid_depth.frag");
    m_SmoothShader = std::make_unique<Shader>("shaders/fluid_quad.vert", "shaders/fluid_smooth.frag");
    m_NormalShader = std::make_unique<Shader>("shaders/fluid_quad.vert", "shaders/fluid_normal.frag");
    m_ThicknessShader = std::make_unique<Shader>("shaders/fluid_thickness.vert", "shaders/fluid_thickness.frag");
    m_ShadingShader = std::make_unique<Shader>("shaders/fluid_quad.vert", "shaders/fluid_shading.frag");
    
    // Setup meshes
    SetupParticleMesh();
    SetupFullscreenQuad();
    
    // Setup framebuffers
    SetupFramebuffers();
}

void FluidRenderer::Shutdown() {
    CleanupFramebuffers();
    
    if (m_ParticleVAO) glDeleteVertexArrays(1, &m_ParticleVAO);
    if (m_ParticleVBO) glDeleteBuffers(1, &m_ParticleVBO);
    if (m_ParticleInstanceVBO) glDeleteBuffers(1, &m_ParticleInstanceVBO);
    if (m_QuadVAO) glDeleteVertexArrays(1, &m_QuadVAO);
    if (m_QuadVBO) glDeleteBuffers(1, &m_QuadVBO);
}

void FluidRenderer::Render(const FluidSimulation* simulation, Camera* camera) {
    if (!simulation) return;
    
    switch (m_RenderMode) {
        case RenderMode::Particles:
            RenderParticles(simulation, camera);
            break;
        case RenderMode::ScreenSpace:
            RenderScreenSpace(simulation, camera);
            break;
    }
}

void FluidRenderer::OnResize(int width, int height) {
    m_ScreenWidth = width;
    m_ScreenHeight = height;
    
    // Recreate framebuffers
    CleanupFramebuffers();
    SetupFramebuffers();
}

void FluidRenderer::RenderParticles(const FluidSimulation* simulation, Camera* camera) {
    const auto& particles = simulation->GetParticles();
    if (particles.empty()) return;
    
    // Update instance data
    UpdateInstanceData(simulation);
    
    // Enable blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    
    // Use particle shader
    m_ParticleShader->Use();
    m_ParticleShader->SetMat4("u_View", camera->GetViewMatrix());
    m_ParticleShader->SetMat4("u_Projection", camera->GetProjectionMatrix());
    m_ParticleShader->SetBool("u_HasTexture", false);
    m_ParticleShader->SetBool("u_SoftParticles", false);
    
    // Render particles
    glBindVertexArray(m_ParticleVAO);
    int activeCount = 0;
    for (const auto& p : particles) {
        if (p.active) activeCount++;
    }
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, activeCount);
    glBindVertexArray(0);
    
    glDisable(GL_BLEND);
}

void FluidRenderer::RenderScreenSpace(const FluidSimulation* simulation, Camera* camera) {
    // TODO: Implement full screen-space rendering pipeline
    // For now, fall back to particle rendering
    RenderParticles(simulation, camera);
}

void FluidRenderer::RenderDepthPass(const FluidSimulation* simulation, Camera* camera) {
    // TODO: Implement depth rendering pass
}

void FluidRenderer::SmoothDepthPass() {
    // TODO: Implement bilateral depth smoothing
}

void FluidRenderer::ComputeNormalsPass() {
    // TODO: Implement normal reconstruction
}

void FluidRenderer::RenderThicknessPass(const FluidSimulation* simulation, Camera* camera) {
    // TODO: Implement thickness calculation
}

void FluidRenderer::FinalShadingPass(Camera* camera) {
    // TODO: Implement final shading with refraction/reflection
}

void FluidRenderer::SetupParticleMesh() {
    // Create quad vertices for billboards
    float quadVertices[] = {
        // Position        
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f, 0.0f
    };
    
    glGenVertexArrays(1, &m_ParticleVAO);
    glGenBuffers(1, &m_ParticleVBO);
    glGenBuffers(1, &m_ParticleInstanceVBO);
    
    glBindVertexArray(m_ParticleVAO);
    
    // Quad vertices
    glBindBuffer(GL_ARRAY_BUFFER, m_ParticleVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    
    // Instance data (will be updated per frame)
    glBindBuffer(GL_ARRAY_BUFFER, m_ParticleInstanceVBO);
    glEnableVertexAttribArray(1);  // Position
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
    glVertexAttribDivisor(1, 1);
    
    glEnableVertexAttribArray(2);  // Color
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glVertexAttribDivisor(2, 1);
    
    glBindVertexArray(0);
}

void FluidRenderer::SetupFullscreenQuad() {
    float quadVertices[] = {
        // Position    // TexCoord
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
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    
    glBindVertexArray(0);
}

void FluidRenderer::SetupFramebuffers() {
    // TODO: Create framebuffers for screen-space rendering
    // For now, create placeholder framebuffers
}

void FluidRenderer::CleanupFramebuffers() {
    if (m_DepthFBO) glDeleteFramebuffers(1, &m_DepthFBO);
    if (m_DepthTexture) glDeleteTextures(1, &m_DepthTexture);
    if (m_SmoothedDepthFBO) glDeleteFramebuffers(1, &m_SmoothedDepthFBO);
    if (m_SmoothedDepthTexture) glDeleteTextures(1, &m_SmoothedDepthTexture);
    if (m_NormalFBO) glDeleteFramebuffers(1, &m_NormalFBO);
    if (m_NormalTexture) glDeleteTextures(1, &m_NormalTexture);
    if (m_ThicknessFBO) glDeleteFramebuffers(1, &m_ThicknessFBO);
    if (m_ThicknessTexture) glDeleteTextures(1, &m_ThicknessTexture);
    
    m_DepthFBO = m_DepthTexture = 0;
    m_SmoothedDepthFBO = m_SmoothedDepthTexture = 0;
    m_NormalFBO = m_NormalTexture = 0;
    m_ThicknessFBO = m_ThicknessTexture = 0;
}

void FluidRenderer::UpdateInstanceData(const FluidSimulation* simulation) {
    const auto& particles = simulation->GetParticles();
    const auto& fluidTypes = simulation->GetFluidType(0);  // Get first fluid type for now
    
    // Prepare instance data
    std::vector<float> instanceData;
    instanceData.reserve(particles.size() * 7);  // 3 pos + 4 color
    
    for (const auto& p : particles) {
        if (!p.active) continue;
        
        // Position
        instanceData.push_back(p.position.x);
        instanceData.push_back(p.position.y);
        instanceData.push_back(p.position.z);
        
        // Color
        instanceData.push_back(p.color.x);
        instanceData.push_back(p.color.y);
        instanceData.push_back(p.color.z);
        instanceData.push_back(0.8f);  // Alpha
    }
    
    // Upload to GPU
    glBindBuffer(GL_ARRAY_BUFFER, m_ParticleInstanceVBO);
    glBufferData(GL_ARRAY_BUFFER, instanceData.size() * sizeof(float), 
                 instanceData.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
