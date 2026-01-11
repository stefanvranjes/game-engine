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
    , m_RenderFoam(true)
    , m_FoamSize(0.015f)
    , m_DepthFBO(0)
    , m_DepthTexture(0)
    , m_SceneDepthTexture(0)
    , m_SmoothedDepthFBO(0)
    , m_SmoothedDepthTexture(0)
    , m_NormalFBO(0)
    , m_NormalTexture(0)
    , m_ThicknessFBO(0)
    , m_ThicknessTexture(0)
    , m_ParticleVAO(0)
    , m_ParticleVBO(0)
    , m_ParticleInstanceVBO(0)
    , m_FoamVAO(0)
    , m_FoamVBO(0)
    , m_FoamInstanceVBO(0)
    , m_FoamTexture(0)
    , m_FoamTextureLoaded(false)
    , m_FoamAnimationTexture(0)
    , m_FoamAnimationTextureLoaded(false)
    , m_FoamNormalTexture(0)
    , m_FoamNormalTextureLoaded(false)
    , m_UseFoamAnimation(false)
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
    m_FoamShader = std::make_unique<Shader>("shaders/foam.vert", "shaders/foam.frag");
    
    // Setup meshes
    SetupParticleMesh();
    SetupFoamMesh();
    SetupFullscreenQuad();
    
    // Setup framebuffers
    SetupFramebuffers();
    
    // Load foam texture
    LoadFoamTexture();
    LoadFoamAnimationTexture();
    LoadFoamNormalTexture();
}

void FluidRenderer::Shutdown() {
    CleanupFramebuffers();
    
    if (m_ParticleVAO) glDeleteVertexArrays(1, &m_ParticleVAO);
    if (m_ParticleVBO) glDeleteBuffers(1, &m_ParticleVBO);
    if (m_ParticleInstanceVBO) glDeleteBuffers(1, &m_ParticleInstanceVBO);
    if (m_FoamVAO) glDeleteVertexArrays(1, &m_FoamVAO);
    if (m_FoamVBO) glDeleteBuffers(1, &m_FoamVBO);
    if (m_FoamInstanceVBO) glDeleteBuffers(1, &m_FoamInstanceVBO);
    if (m_FoamTexture) glDeleteTextures(1, &m_FoamTexture);
    if (m_FoamAnimationTexture) glDeleteTextures(1, &m_FoamAnimationTexture);
    if (m_FoamNormalTexture) glDeleteTextures(1, &m_FoamNormalTexture);
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
    
    // Render foam particles (always on top)
    if (m_RenderFoam) {
        RenderFoamParticles(simulation, camera);
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
    const auto& particles = simulation->GetParticles();
    if (particles.empty()) return;
    
    // Update instance data
    UpdateInstanceData(simulation);
    
    // 1. Render depth pass
    RenderDepthPass(simulation, camera);
    
    // 2. Smooth depth
    SmoothDepthPass();
    
    // 3. Compute normals
    ComputeNormalsPass(camera);
    
    // 4. Render thickness
    RenderThicknessPass(simulation, camera);
    
    // 5. Final shading
    FinalShadingPass(camera);
}

void FluidRenderer::RenderDepthPass(const FluidSimulation* simulation, Camera* camera) {
    const auto& particles = simulation->GetParticles();
    
    // Bind depth framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, m_DepthFBO);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    
    // Use depth shader
    m_DepthShader->Use();
    m_DepthShader->SetMat4("u_View", camera->GetViewMatrix());
    m_DepthShader->SetMat4("u_Projection", camera->GetProjectionMatrix());
    m_DepthShader->SetFloat("u_ParticleSizeScale", m_ParticleSize);
    m_DepthShader->SetFloat("u_ParticleRadius", m_ParticleSize);
    
    // Render particles
    glBindVertexArray(m_ParticleVAO);
    int activeCount = 0;
    for (const auto& p : particles) {
        if (p.active) activeCount++;
    }
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, activeCount);
    glBindVertexArray(0);
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FluidRenderer::SmoothDepthPass() {
    // Bind smoothed depth framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, m_SmoothedDepthFBO);
    glClear(GL_COLOR_BUFFER_BIT);
    
    glDisable(GL_DEPTH_TEST);
    
    // Use smooth shader
    m_SmoothShader->Use();
    m_SmoothShader->SetInt("u_DepthTexture", 0);
    m_SmoothShader->SetVec2("u_TexelSize", Vec3(1.0f / m_ScreenWidth, 1.0f / m_ScreenHeight, 0.0f));
    m_SmoothShader->SetFloat("u_FilterRadius", m_SmoothingRadius);
    m_SmoothShader->SetFloat("u_DepthThreshold", 0.1f);
    
    // Bind depth texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_DepthTexture);
    
    // Render fullscreen quad
    glBindVertexArray(m_QuadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FluidRenderer::ComputeNormalsPass(Camera* camera) {
    // Bind normal framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, m_NormalFBO);
    glClear(GL_COLOR_BUFFER_BIT);
    
    glDisable(GL_DEPTH_TEST);
    
    // Use normal shader
    m_NormalShader->Use();
    m_NormalShader->SetInt("u_DepthTexture", 0);
    m_NormalShader->SetVec2("u_TexelSize", Vec3(1.0f / m_ScreenWidth, 1.0f / m_ScreenHeight, 0.0f));
    m_NormalShader->SetMat4("u_Projection", camera->GetProjectionMatrix());
    
    // Bind smoothed depth texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_SmoothedDepthTexture);
    
    // Render fullscreen quad
    glBindVertexArray(m_QuadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FluidRenderer::RenderThicknessPass(const FluidSimulation* simulation, Camera* camera) {
    const auto& particles = simulation->GetParticles();
    
    // Bind thickness framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, m_ThicknessFBO);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Enable additive blending for thickness accumulation
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glDisable(GL_DEPTH_TEST);
    
    // Use thickness shader
    m_ThicknessShader->Use();
    m_ThicknessShader->SetMat4("u_View", camera->GetViewMatrix());
    m_ThicknessShader->SetMat4("u_Projection", camera->GetProjectionMatrix());
    m_ThicknessShader->SetFloat("u_ParticleSizeScale", m_ParticleSize);
    m_ThicknessShader->SetFloat("u_ThicknessScale", m_ThicknessScale);
    
    // Render particles
    glBindVertexArray(m_ParticleVAO);
    int activeCount = 0;
    for (const auto& p : particles) {
        if (p.active) activeCount++;
    }
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, activeCount);
    glBindVertexArray(0);
    
    glDisable(GL_BLEND);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FluidRenderer::FinalShadingPass(Camera* camera) {
    // Render to default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_DEPTH_TEST);
    
    // Use shading shader
    m_ShadingShader->Use();
    m_ShadingShader->SetInt("u_DepthTexture", 0);
    m_ShadingShader->SetInt("u_NormalTexture", 1);
    m_ShadingShader->SetInt("u_ThicknessTexture", 2);
    m_ShadingShader->SetInt("u_SceneTexture", 3);  // TODO: Bind actual scene texture
    m_ShadingShader->SetInt("u_SceneDepthTexture", 4);  // TODO: Bind actual scene depth
    
    m_ShadingShader->SetMat4("u_View", camera->GetViewMatrix());
    m_ShadingShader->SetMat4("u_Projection", camera->GetProjectionMatrix());
    
    // Calculate inverse projection
    auto proj = camera->GetProjectionMatrix();
    // TODO: Implement matrix inverse
    m_ShadingShader->SetMat4("u_InvProjection", proj);  // Placeholder
    
    m_ShadingShader->SetVec3("u_CameraPos", camera->GetPosition());
    m_ShadingShader->SetVec3("u_LightDir", Vec3(0.0f, -1.0f, 0.0f).Normalized());
    
    m_ShadingShader->SetVec3("u_FluidColor", Vec3(0.2f, 0.5f, 1.0f));
    m_ShadingShader->SetFloat("u_RefractiveIndex", m_RefractiveIndex);
    m_ShadingShader->SetVec3("u_AbsorptionColor", m_AbsorptionColor);
    m_ShadingShader->SetFloat("u_FresnelPower", m_FresnelPower);
    m_ShadingShader->SetFloat("u_SpecularPower", 32.0f);
    
    // Bind textures
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_SmoothedDepthTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_NormalTexture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, m_ThicknessTexture);
    // TODO: Bind scene texture and depth
    
    // Render fullscreen quad
    glBindVertexArray(m_QuadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
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
    // Create depth framebuffer and texture
    glGenFramebuffers(1, &m_DepthFBO);
    glGenTextures(1, &m_DepthTexture);
    
    glBindTexture(GL_TEXTURE_2D, m_DepthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, m_ScreenWidth, m_ScreenHeight, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glBindFramebuffer(GL_FRAMEBUFFER, m_DepthFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_DepthTexture, 0);
    
    // Create depth renderbuffer for depth testing
    unsigned int depthRBO;
    glGenRenderbuffers(1, &depthRBO);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, m_ScreenWidth, m_ScreenHeight);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRBO);
    
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        // Error handling
    }
    
    // Create smoothed depth framebuffer
    glGenFramebuffers(1, &m_SmoothedDepthFBO);
    glGenTextures(1, &m_SmoothedDepthTexture);
    
    glBindTexture(GL_TEXTURE_2D, m_SmoothedDepthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, m_ScreenWidth, m_ScreenHeight, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glBindFramebuffer(GL_FRAMEBUFFER, m_SmoothedDepthFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_SmoothedDepthTexture, 0);
    
    // Create normal framebuffer
    glGenFramebuffers(1, &m_NormalFBO);
    glGenTextures(1, &m_NormalTexture);
    
    glBindTexture(GL_TEXTURE_2D, m_NormalTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, m_ScreenWidth, m_ScreenHeight, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glBindFramebuffer(GL_FRAMEBUFFER, m_NormalFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_NormalTexture, 0);
    
    // Create thickness framebuffer
    glGenFramebuffers(1, &m_ThicknessFBO);
    glGenTextures(1, &m_ThicknessTexture);
    
    glBindTexture(GL_TEXTURE_2D, m_ThicknessTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, m_ScreenWidth, m_ScreenHeight, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glBindFramebuffer(GL_FRAMEBUFFER, m_ThicknessFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_ThicknessTexture, 0);
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
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

void FluidRenderer::RenderFoamParticles(const FluidSimulation* simulation, Camera* camera) {
    if (!simulation->GetFoamSystem()) return;
    
    const auto& foamParticles = simulation->GetFoamSystem()->GetParticles();
    if (foamParticles.empty()) return;
    
    // Prepare instance data for foam particles
    std::vector<float> instanceData;
    instanceData.reserve(foamParticles.size() * 13);  // 3 pos + 4 color + 1 size + 1 textureIndex + 1 animationTime + 3 velocity
    
    for (const auto& p : foamParticles) {
        if (!p.active) continue;
        
        // Position
        instanceData.push_back(p.position.x);
        instanceData.push_back(p.position.y);
        instanceData.push_back(p.position.z);
        
        // Color based on particle type and lifetime
        float lifeRatio = p.GetLifeRatio();
        Vec3 color;
        float alpha;
        
        switch (p.type) {
            case FoamParticle::Type::Foam:
                // White foam
                color = Vec3(1.0f, 1.0f, 1.0f);
                alpha = 0.8f * lifeRatio;
                break;
            case FoamParticle::Type::Spray:
                // Slightly blue spray
                color = Vec3(0.9f, 0.95f, 1.0f);
                alpha = 0.6f * lifeRatio;
                break;
            case FoamParticle::Type::Bubble:
                // Transparent bubbles
                color = Vec3(0.95f, 0.98f, 1.0f);
                alpha = 0.4f * lifeRatio;
                break;
        }
        
        instanceData.push_back(color.x);
        instanceData.push_back(color.y);
        instanceData.push_back(color.z);
        instanceData.push_back(alpha);
        
        // Size (scaled by lifetime)
        instanceData.push_back(p.size);
        
        // Texture index
        instanceData.push_back(static_cast<float>(p.textureIndex));
        
        // Animation time
        instanceData.push_back(p.animationTime);

        // Velocity for flow distortion
        instanceData.push_back(p.velocity.x);
        instanceData.push_back(p.velocity.y);
        instanceData.push_back(p.velocity.z);
    }
    
    if (instanceData.empty()) return;
    
    // Upload to GPU
    glBindBuffer(GL_ARRAY_BUFFER, m_FoamInstanceVBO);
    glBufferData(GL_ARRAY_BUFFER, instanceData.size() * sizeof(float), 
                 instanceData.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    // Enable blending for foam
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);  // Don't write to depth buffer
    
    // Use foam shader
    m_FoamShader->Use();
    m_FoamShader->SetMat4("u_View", camera->GetViewMatrix());
    m_FoamShader->SetMat4("u_Projection", camera->GetProjectionMatrix());
    
    // Configure animation
    m_FoamShader->SetBool("u_UseAnimation", m_UseFoamAnimation);
    m_FoamShader->SetInt("u_AnimationFrames", 16);  // 4x4 atlas
    
    // Bind appropriate texture
    // Bind appropriate texture
    if (m_UseFoamAnimation && m_FoamAnimationTextureLoaded) {
        m_FoamShader->SetBool("u_HasTexture", true);
        m_FoamShader->SetInt("u_FoamTexture", 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_FoamAnimationTexture);
        
        // Bind normal map if available (only for animation)
        if (m_FoamNormalTextureLoaded) {
            m_FoamShader->SetBool("u_HasNormalMap", true);
            m_FoamShader->SetInt("u_FoamNormalMap", 1);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, m_FoamNormalTexture);
        } else {
            m_FoamShader->SetBool("u_HasNormalMap", false);
        }
    } 
    else if (m_FoamTextureLoaded) {
        m_FoamShader->SetBool("u_HasTexture", true);
        m_FoamShader->SetInt("u_FoamTexture", 0);
        m_FoamShader->SetBool("u_HasNormalMap", false); // No normal map for static atlas yet
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_FoamTexture);
    } else {
        m_FoamShader->SetBool("u_HasTexture", false);
        m_FoamShader->SetBool("u_HasNormalMap", false);
    }
    
    // Set lighting parameters
    m_FoamShader->SetVec3("u_LightDir", Vec3(0.5f, 1.0f, 0.2f).Normalized()); // Hardcoded light for now
    m_FoamShader->SetVec3("u_LightColor", Vec3(1.0f, 1.0f, 1.0f));
    m_FoamShader->SetVec3("u_AmbientColor", Vec3(0.4f, 0.4f, 0.5f));
    
    // Soft particles
    if (m_SceneDepthTexture != 0) {
        m_FoamShader->SetBool("u_UseSoftParticles", true);
        m_FoamShader->SetInt("u_SceneDepth", 2); // Texture unit 2
        m_FoamShader->SetVec2("u_ScreenSize", Vec2((float)m_ScreenWidth, (float)m_ScreenHeight));
        m_FoamShader->SetVec2("u_CameraNearFar", Vec2(camera->GetNearPlane(), camera->GetFarPlane()));
        
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, m_SceneDepthTexture);
    } else {
        m_FoamShader->SetBool("u_UseSoftParticles", false);
    }
    
    // Render foam particles
    // Render foam particles
    glBindVertexArray(m_FoamVAO);
    int activeCount = static_cast<int>(instanceData.size() / 13);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, activeCount);
    glBindVertexArray(0);
    
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

void FluidRenderer::SetupFoamMesh() {
    // Create quad vertices for billboards
    float quadVertices[] = {
        // Position        
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f, 0.0f
    };
    
    glGenVertexArrays(1, &m_FoamVAO);
    glGenBuffers(1, &m_FoamVBO);
    glGenBuffers(1, &m_FoamInstanceVBO);
    
    glBindVertexArray(m_FoamVAO);
    
    // Quad vertices
    glBindBuffer(GL_ARRAY_BUFFER, m_FoamVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    
    // Instance data (will be updated per frame)
    // Layout: 3 pos + 4 color + 1 size + 1 textureIndex + 1 animationTime + 3 velocity = 13 floats
    glBindBuffer(GL_ARRAY_BUFFER, m_FoamInstanceVBO);
    
    glEnableVertexAttribArray(1);  // Position
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 13 * sizeof(float), (void*)0);
    glVertexAttribDivisor(1, 1);
    
    glEnableVertexAttribArray(2);  // Color
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 13 * sizeof(float), (void*)(3 * sizeof(float)));
    glVertexAttribDivisor(2, 1);
    
    glEnableVertexAttribArray(3);  // Size
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 13 * sizeof(float), (void*)(7 * sizeof(float)));
    glVertexAttribDivisor(3, 1);
    
    glEnableVertexAttribArray(4);  // Texture index
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 13 * sizeof(float), (void*)(8 * sizeof(float)));
    glVertexAttribDivisor(4, 1);
    
    glEnableVertexAttribArray(5);  // Animation time
    glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, 13 * sizeof(float), (void*)(9 * sizeof(float)));
    glVertexAttribDivisor(5, 1);

    glEnableVertexAttribArray(6);  // Velocity
    glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, 13 * sizeof(float), (void*)(10 * sizeof(float)));
    glVertexAttribDivisor(6, 1);
    
    glBindVertexArray(0);
}

void FluidRenderer::LoadFoamTexture() {
    // Load foam texture atlas
    // This is a placeholder - actual texture loading would use stb_image or similar
    glGenTextures(1, &m_FoamTexture);
    glBindTexture(GL_TEXTURE_2D, m_FoamTexture);
    
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // TODO: Load actual texture from file using stb_image
    // For now, create a simple procedural texture
    const int size = 512;
    std::vector<unsigned char> textureData(size * size * 4, 255);
    
    // Create simple gradient circles for each atlas cell
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int cellX = x / (size / 2);
            int cellY = y / (size / 2);
            
            float localX = (x % (size / 2)) / (float)(size / 2);
            float localY = (y % (size / 2)) / (float)(size / 2);
            
            float centerX = 0.5f;
            float centerY = 0.5f;
            float dist = std::sqrt((localX - centerX) * (localX - centerX) + 
                                  (localY - centerY) * (localY - centerY));
            
            int idx = (y * size + x) * 4;
            float alpha = 1.0f - std::min(dist * 2.0f, 1.0f);
            textureData[idx + 3] = static_cast<unsigned char>(alpha * 255);
        }
    }
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    
    m_FoamTextureLoaded = true;
    glBindTexture(GL_TEXTURE_2D, 0);
}

void FluidRenderer::LoadFoamNormalTexture() {
    // Load foam normal map texture
    glGenTextures(1, &m_FoamNormalTexture);
    glBindTexture(GL_TEXTURE_2D, m_FoamNormalTexture);
    
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // Placeholder procedural normal map (flat normals)
    const int size = 512;
    std::vector<unsigned char> textureData(size * size * 4, 128); // 128 ~ 0.5 in 0-1 range
    
    // 4x4 grid = 16 frames
    int framesPerRow = 4;
    int cellSize = size / framesPerRow;
    
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int idx = (y * size + x) * 4;
            
            // Default flat normal (0.5, 0.5, 1.0) -> (128, 128, 255)
            textureData[idx + 0] = 128; // X
            textureData[idx + 1] = 128; // Y
            textureData[idx + 2] = 255; // Z
            textureData[idx + 3] = 255; // Alpha
            
            // Generate some spherical normals for bubbles
            int cellX = x / cellSize;
            int cellY = y / cellSize;
            
            float localX = (x % cellSize) / (float)cellSize;
            float localY = (y % cellSize) / (float)cellSize;
            float centerX = 0.5f;
            float centerY = 0.5f;
            float dist = std::sqrt((localX - centerX) * (localX - centerX) + 
                                  (localY - centerY) * (localY - centerY));
                                  
            if (dist < 0.45f) {
                // Sphere normal
                float nx = (localX - centerX) / 0.45f;
                float ny = (localY - centerY) / 0.45f;
                float nz = std::sqrt(std::max(0.0f, 1.0f - nx*nx - ny*ny));
                
                // Map -1..1 to 0..255
                textureData[idx + 0] = static_cast<unsigned char>((nx * 0.5f + 0.5f) * 255);
                textureData[idx + 1] = static_cast<unsigned char>((ny * 0.5f + 0.5f) * 255);
                textureData[idx + 2] = static_cast<unsigned char>((nz * 0.5f + 0.5f) * 255);
            }
        }
    }
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    
    m_FoamNormalTextureLoaded = true;
    glBindTexture(GL_TEXTURE_2D, 0);
}

void FluidRenderer::LoadFoamAnimationTexture() {
    // Load foam animation texture (atlas)
    // We use a procedural fallback if the file is not found
    
    // HDR Texture: Use GL_RGBA16F for high dynamic range values
    glGenTextures(1, &m_FoamAnimationTexture);
    glBindTexture(GL_TEXTURE_2D, m_FoamAnimationTexture);
    
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // Placeholder procedural animation texture with HDR sparkles
    const int size = 512;
    // Use float data for HDR
    std::vector<float> textureData(size * size * 4, 1.0f); 
    
    // 4x4 grid = 16 frames
    int framesPerRow = 4;
    int cellSize = size / framesPerRow;
    
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int cellX = x / cellSize;
            int cellY = y / cellSize;
            int frameIndex = cellY * framesPerRow + cellX;
            
            float localX = (x % cellSize) / (float)cellSize;
            float localY = (y % cellSize) / (float)cellSize;
            
            float centerX = 0.5f;
            float centerY = 0.5f;
            float dist = std::sqrt((localX - centerX) * (localX - centerX) + 
                                  (localY - centerY) * (localY - centerY));
            
            // Animation: expanding and fading circle
            float t = frameIndex / 16.0f; // 0 to 1
            float radius = 0.2f + 0.3f * std::sin(t * 3.14159f);
            float opacity = 1.0f;
            
            if (t > 0.8f) opacity = 1.0f - (t - 0.8f) * 5.0f; // Fade out at end
            
            int idx = (y * size + x) * 4;
            float alpha = 1.0f - std::min(dist / radius, 1.0f);
            alpha = std::pow(alpha, 3.0f) * opacity;
            
            // Base white color
            textureData[idx + 0] = 1.0f;
            textureData[idx + 1] = 1.0f;
            textureData[idx + 2] = 1.0f;
            textureData[idx + 3] = alpha;
            
            // Add HDR sparkles in the center
            if (dist < radius * 0.3f && alpha > 0.1f) {
                // Random sparkle intensity based on position hash
                float noise = std::fmod(std::sin(x * 12.9898f + y * 78.233f) * 43758.5453f, 1.0f);
                if (noise > 0.95f) {
                    float intensity = 5.0f + noise * 5.0f; // 5.0 to 10.0 intensity (HDR!)
                    textureData[idx + 0] *= intensity;
                    textureData[idx + 1] *= intensity;
                    textureData[idx + 2] *= intensity;
                }
            }
        }
    }
    
    // Use GL_RGBA16F internal format and GL_FLOAT type
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, size, size, 0, GL_RGBA, GL_FLOAT, textureData.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    
    m_FoamAnimationTextureLoaded = true;
    glBindTexture(GL_TEXTURE_2D, 0);
}
