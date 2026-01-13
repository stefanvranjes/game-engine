#include "ParticleSystem.h"
#include "GBuffer.h"
#include "GLExtensions.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <algorithm>

ParticleSystem::ParticleSystem()
    : m_QuadVAO(0)
    , m_QuadVBO(0)
    , m_InstanceVBO(0)
    , m_TrailVAO(0)
    , m_TrailVBO(0)
    , m_TrailEBO(0)
{
}

ParticleSystem::~ParticleSystem() {
    Shutdown();
}

bool ParticleSystem::Init() {
    // Load particle shader
    m_Shader = std::make_unique<Shader>();
    if (!m_Shader->LoadFromFiles("shaders/particle.vert", "shaders/particle.frag")) {
        std::cerr << "Failed to load particle shader" << std::endl;
        return false;
    }
    
    // Load trail shader
    m_TrailShader = std::make_unique<Shader>();
    if (!m_TrailShader->LoadFromFiles("shaders/trail.vert", "shaders/trail.frag")) {
        std::cerr << "Failed to load trail shader" << std::endl;
        return false;
    }

    SetupQuadMesh();
    
    // Setup trail buffers
    glGenVertexArrays(1, &m_TrailVAO);
    glGenBuffers(1, &m_TrailVBO);
    glGenBuffers(1, &m_TrailEBO);
    
    std::cout << "ParticleSystem initialized" << std::endl;
    return true;
}

void ParticleSystem::SetupQuadMesh() {
    // Create a simple quad (2 triangles)
    float quadVertices[] = {
        // Positions     // TexCoords
        -0.5f,  0.5f, 0.0f,  0.0f, 1.0f,
        -0.5f, -0.5f, 0.0f,  0.0f, 0.0f,
         0.5f, -0.5f, 0.0f,  1.0f, 0.0f,
         
        -0.5f,  0.5f, 0.0f,  0.0f, 1.0f,
         0.5f, -0.5f, 0.0f,  1.0f, 0.0f,
         0.5f,  0.5f, 0.0f,  1.0f, 1.0f
    };

    glGenVertexArrays(1, &m_QuadVAO);
    glGenBuffers(1, &m_QuadVBO);
    glGenBuffers(1, &m_InstanceVBO);

    glBindVertexArray(m_QuadVAO);

    // Quad vertices
    glBindBuffer(GL_ARRAY_BUFFER, m_QuadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    // Position attribute
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);

    // TexCoord attribute
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

    // Instance buffer (will be filled per frame)
    glBindBuffer(GL_ARRAY_BUFFER, m_InstanceVBO);

    // Particle position (instanced)
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(ParticleSystem::InstanceData), (void*)offsetof(ParticleSystem::InstanceData, position));
    glVertexAttribDivisor(2, 1);

    // Particle color (instanced)
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(ParticleSystem::InstanceData), (void*)offsetof(ParticleSystem::InstanceData, color));
    glVertexAttribDivisor(3, 1);

    // Particle size (instanced)
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(ParticleSystem::InstanceData), (void*)offsetof(ParticleSystem::InstanceData, size));
    glVertexAttribDivisor(4, 1);

    // Particle lifeRatio (instanced)
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, sizeof(ParticleSystem::InstanceData), (void*)offsetof(ParticleSystem::InstanceData, lifeRatio));
    glVertexAttribDivisor(5, 1);

    glBindVertexArray(0);
}

void ParticleSystem::Update(float deltaTime, const Vec3& cameraPos) {
    // 1. Calculate Total Active Particles
    m_TotalActiveParticles = 0;
    for (const auto& emitter : m_Emitters) {
        if (emitter->IsActive()) {
            m_TotalActiveParticles += emitter->GetActiveParticleCount();
        }
    }
    
    // 2. Budget Check & Stealing Logic
    // If Over Budget
    if (m_TotalActiveParticles > m_GlobalParticleLimit) {
        int overBudget = m_TotalActiveParticles - m_GlobalParticleLimit;
        
        // Find candidates to kill (Low priority)
        // Sort emitters by priority (Ascending) to find cheapest victims
        // We don't want to sort the main list every frame.
        // Let's just iterate.
        
        for (auto& emitter : m_Emitters) {
            if (overBudget <= 0) break;
            if (!emitter->IsActive()) continue;
            
            // If Emitter is Low Priority (e.g. < 5) and has particles
            if (emitter->GetPriority() < 5 && emitter->GetActiveParticleCount() > 0) {
                // Kill some!
                int killed = emitter->KillOldest(10); // Kill 10 at a time
                overBudget -= killed;
                m_TotalActiveParticles -= killed;
            }
        }
    }
    
    // 3. Update Emitters
    
    // Prepare Physics Context - DISABLED
    // ParticleEmitter::PhysicsContext physCtx;
    // physCtx.wind = m_GlobalWind;
    // physCtx.attractors = &m_Attractors;

    for (auto& emitter : m_Emitters) {
        // Enforce limit on spawning logic inside emitter?
        // Ideally we pass "CanSpawn" flag or "BudgetRemaining"
        // But ParticleEmitter::Update controls spawning independently.
        // For now, if we are over budget, we just rely on KillOldest above to balance it out slowly,
        // or we could disable spawning for low priority emitters here.
        
        if (m_TotalActiveParticles > m_GlobalParticleLimit && emitter->GetPriority() < 5) {
             // Skip update? No, existing particles need to move.
             // We need to tell it to STOP SPAWNING.
             // Simple hack: Set SpawnRate to 0 temporarily? No, that messes up state.
             // Let's assume the budget logic above is sufficient for now.
             // Or better: Add a "SetPaused(bool)" or separate UpdateSimulation vs UpdateSpawning.
             // For now, just Update.
        }
        
        // emitter->SetPhysicsContext(physCtx); // DISABLED
        emitter->Update(deltaTime, cameraPos);
    }
}

void ParticleSystem::Render(Camera* camera, GBuffer* gbuffer) {
    if (!camera || !m_Shader) return;

    // Render particles grouped by emitter (for texture/blend mode batching)
    m_Shader->Use();
    m_Shader->SetMat4("u_View", camera->GetViewMatrix().m);
    m_Shader->SetMat4("u_Projection", camera->GetProjectionMatrix().m);
    
    // Soft particles setup
    if (gbuffer) {
        // Get screen size
        int width, height;
        glfwGetFramebufferSize(glfwGetCurrentContext(), &width, &height);
        
        // Bind depth texture
        glActiveTexture(GL_TEXTURE1);
        // Bind depth texture
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, gbuffer->GetDepthTexture());
        
        m_Shader->SetInt("u_DepthTexture", 1);
        m_Shader->SetInt("u_SoftParticles", 1);
        m_Shader->SetVec2("u_ScreenSize", static_cast<float>(width), static_cast<float>(height));
        m_Shader->SetFloat("u_Softness", 0.1f); // Fade distance in depth units
    } else {
        m_Shader->SetInt("u_SoftParticles", 0);
    }

    glDepthMask(GL_FALSE); // Don't write to depth buffer
    glEnable(GL_BLEND);

    for (auto& emitter : m_Emitters) {
        // Check if using persistent GPU mode
        if (emitter->GetGPUPersistent() && emitter->GetUseGPUCompute()) {
            RenderFromGPU(emitter, camera);
            continue;
        }
    
        // Collect active particles from this emitter
        m_InstanceData.clear();
        const auto& particles = emitter->GetParticles();
        
        for (const auto& particle : particles) {
            if (particle.active) {
                InstanceData data;
                data.position[0] = particle.position.x;
                data.position[1] = particle.position.y;
                data.position[2] = particle.position.z;
                data.color[0] = particle.color.x;
                data.color[1] = particle.color.y;
                data.color[2] = particle.color.z;
                data.color[3] = particle.color.w;
                data.size = particle.size;
                data.lifeRatio = particle.lifeRatio;
                m_InstanceData.push_back(data);
            }
        }

        if (m_InstanceData.empty()) continue;

        // Set blend mode
        if (emitter->GetBlendMode() == BlendMode::Additive) {
            glBlendFunc(GL_SRC_ALPHA, GL_ONE); // Additive blending
        } else {
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // Alpha blending
        }

        // Bind texture if available
        auto texture = emitter->GetTexture();
        if (texture) {
            texture->Bind(0);
            m_Shader->SetInt("u_Texture", 0);
            m_Shader->SetInt("u_HasTexture", 1);
        } else {
            m_Shader->SetInt("u_HasTexture", 0);
        }

        // Update instance buffer
        glBindBuffer(GL_ARRAY_BUFFER, m_InstanceVBO);
        glBufferData(GL_ARRAY_BUFFER, m_InstanceData.size() * sizeof(InstanceData), m_InstanceData.data(), GL_DYNAMIC_DRAW);

        // Draw instances
        glBindVertexArray(m_QuadVAO);
        
        // Set atlas uniforms
        m_Shader->SetFloat("u_AtlasRows", static_cast<float>(emitter->GetAtlasRows()));
        m_Shader->SetFloat("u_AtlasCols", static_cast<float>(emitter->GetAtlasCols()));
        
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, static_cast<GLsizei>(m_InstanceData.size()));
    }

    glBindVertexArray(0);
    
    // Render trails
    RenderTrails(camera);

    // Restore state
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}


void ParticleSystem::SortParticlesGPU(const std::shared_ptr<ParticleEmitter>& emitter, Camera* camera) {
    if (!m_SortInitShader || !m_SortStepShader) return;
    
    unsigned int particleSSBO = emitter->GetParticleSSBO();
    unsigned int sortSSBO = emitter->GetSortSSBO();
    unsigned int atomicCounter = emitter->GetAtomicCounterBuffer();
    unsigned int sortBufferSize = emitter->GetSortBufferSize();
    
    if (sortSSBO == 0 || particleSSBO == 0) return;

    // 1. Initialize Sort Buffer
    m_SortInitShader->Use();
    m_SortInitShader->SetVec3("u_CameraPosition", camera->GetPosition().x, camera->GetPosition().y, camera->GetPosition().z);
    m_SortInitShader->SetInt("u_MaxParticles", static_cast<int>(emitter->GetParticles().size())); 
    m_SortInitShader->SetInt("u_SortBufferSize", static_cast<int>(sortBufferSize));
    
    
    // 2. Bitonic Sort
    m_SortStepShader->Use();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, sortSSBO);
    
    unsigned int ssbo = emitter->GetParticleSSBO();
    unsigned int activeCount = emitter->GetActiveParticleCount();
    
    if (ssbo == 0 || activeCount == 0) return;
    
    // Sort particles
    SortParticlesGPU(emitter, camera);
    
    m_GPUShader->Use();
    m_GPUShader->SetMat4("u_View", camera->GetViewMatrix().m);
    m_GPUShader->SetMat4("u_Projection", camera->GetProjectionMatrix().m);
    
    // Set blend mode
    if (emitter->GetBlendMode() == BlendMode::Additive) {
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    } else {
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }
    
    // Bind texture
    auto texture = emitter->GetTexture();
    if (texture) {
        texture->Bind(0);
        m_GPUShader->SetInt("u_Texture", 0);
        m_GPUShader->SetInt("u_HasTexture", 1);
    } else {
        m_GPUShader->SetInt("u_HasTexture", 0);
    }
    
    // Set atlas uniforms
    m_GPUShader->SetFloat("u_AtlasRows", static_cast<float>(emitter->GetAtlasRows()));
    m_GPUShader->SetFloat("u_AtlasCols", static_cast<float>(emitter->GetAtlasCols()));
    
    // Bind SSBOs
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    if (sortSSBO != 0) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, sortSSBO);
        m_GPUShader->SetInt("u_UseSort", 1);
    } else {
        m_GPUShader->SetInt("u_UseSort", 0);
    }
    
    
    // Draw using vertex pulling
    glBindVertexArray(m_QuadVAO); 
    
    // Check for Indirect Draw - DISABLED (missing GL extensions)
    // unsigned int indirectBuffer = emitter->GetIndirectBuffer();
    // if (indirectBuffer != 0) {
    //     glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirectBuffer);
    //     glDrawArraysIndirect(GL_TRIANGLES, 0);
    //     glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    // } else {
        // Fallback to CPU readback
        // (activeCount was readback earlier in UpdateGPU if fallback)
        glDrawArrays(GL_TRIANGLES, 0, activeCount * 6);
    // }
    
    glBindVertexArray(0);
}

void ParticleSystem::Shutdown() {
    if (m_QuadVAO != 0) {
        glDeleteVertexArrays(1, &m_QuadVAO);
        m_QuadVAO = 0;
    }
    if (m_QuadVBO != 0) {
        glDeleteBuffers(1, &m_QuadVBO);
        m_QuadVBO = 0;
    }
    if (m_InstanceVBO != 0) {
        glDeleteBuffers(1, &m_InstanceVBO);
        m_InstanceVBO = 0;
    }
    if (m_TrailVAO != 0) {
        glDeleteVertexArrays(1, &m_TrailVAO);
        m_TrailVAO = 0;
    }
    if (m_TrailVBO != 0) {
        glDeleteBuffers(1, &m_TrailVBO);
        m_TrailVBO = 0;
    }
    if (m_TrailEBO != 0) {
        glDeleteBuffers(1, &m_TrailEBO);
        m_TrailEBO = 0;
    }
}

void ParticleSystem::AddEmitter(std::shared_ptr<ParticleEmitter> emitter) {
    m_Emitters.push_back(emitter);
}

void ParticleSystem::RemoveEmitter(std::shared_ptr<ParticleEmitter> emitter) {
    m_Emitters.erase(
        std::remove(m_Emitters.begin(), m_Emitters.end(), emitter),
        m_Emitters.end()
    );
}

void ParticleSystem::ClearEmitters() {
    m_Emitters.clear();
}

void ParticleSystem::RenderTrails(Camera* camera) {
    if (!camera || !m_TrailShader) return;
    
    // Disable depth write for transparent trails
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // Default, can be overridden per emitter
    
    for (const auto& emitter : m_Emitters) {
        if (!emitter->IsActive() || !emitter->GetEnableTrails()) continue;
        
        if (emitter->GetUseGPUCompute()) {
            RenderTrailsGPU(emitter, camera);
        }
    }
    
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}
void ParticleSystem::RenderFromGPU(const std::shared_ptr<ParticleEmitter>& emitter, Camera* camera) {
    if (!emitter || !camera || !m_GPUShader) return;
    
    // Sort particles on GPU before rendering
    SortParticlesGPU(emitter, camera);
    
    m_GPUShader->Use();
    m_GPUShader->SetMat4("u_View", camera->GetViewMatrix().m);
    m_GPUShader->SetMat4("u_Projection", camera->GetProjectionMatrix().m);
    
    // Set blend mode
    if (emitter->GetBlendMode() == BlendMode::Additive) {
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    } else {
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }
    
    // Bind texture
    auto texture = emitter->GetTexture();
    if (texture) {
        texture->Bind(0);
        m_GPUShader->SetInt("u_Texture", 0);
        m_GPUShader->SetInt("u_HasTexture", 1);
    } else {
        m_GPUShader->SetInt("u_HasTexture", 0);
    }
    
    // Set atlas uniforms
    m_GPUShader->SetFloat("u_AtlasRows", static_cast<float>(emitter->GetAtlasRows()));
    m_GPUShader->SetFloat("u_AtlasCols", static_cast<float>(emitter->GetAtlasCols()));
    
    // Bind SSBO
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, emitter->GetParticleSSBO());
    
    // Draw using vertex pulling
    glBindVertexArray(m_QuadVAO);
    glDrawArrays(GL_TRIANGLES, 0, emitter->GetActiveParticleCount() * 6);
    glBindVertexArray(0);
}

void ParticleSystem::RenderTrailsGPU(const std::shared_ptr<ParticleEmitter>& emitter, Camera* camera) {
    if (!emitter || !camera || !m_TrailShader) return;
    
    // GPU Trails implementation would go here
    // For now, this is a placeholder to resolve linker errors
}
