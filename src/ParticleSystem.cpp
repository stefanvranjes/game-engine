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

    SetupQuadMesh();
    
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
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (void*)offsetof(InstanceData, position));
    glVertexAttribDivisor(2, 1);

    // Particle color (instanced)
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (void*)offsetof(InstanceData, color));
    glVertexAttribDivisor(3, 1);

    // Particle size (instanced)
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (void*)offsetof(InstanceData, size));
    glVertexAttribDivisor(4, 1);

    glBindVertexArray(0);
}

void ParticleSystem::Update(float deltaTime) {
    for (auto& emitter : m_Emitters) {
        emitter->Update(deltaTime);
    }
}

void ParticleSystem::Render(Camera* camera, GBuffer* gbuffer) {
    if (!camera || !m_Shader) return;

    // Render particles grouped by emitter (for texture/blend mode batching)
    m_Shader->Use();
    m_Shader->SetMat4("u_View", camera->GetViewMatrix().GetData());
    m_Shader->SetMat4("u_Projection", camera->GetProjectionMatrix().GetData());
    
    // Soft particles setup
    if (gbuffer) {
        // Get screen size
        int width, height;
        glfwGetFramebufferSize(glfwGetCurrentContext(), &width, &height);
        
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
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, static_cast<GLsizei>(m_InstanceData.size()));
    }

    glBindVertexArray(0);

    // Restore state
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
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
