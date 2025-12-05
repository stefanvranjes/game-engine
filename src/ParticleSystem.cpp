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
    
    // Load GPU particle shader (vertex pulling)
    m_GPUShader = std::make_unique<Shader>();
    if (!m_GPUShader->LoadFromFiles("shaders/particle_gpu.vert", "shaders/particle.frag")) {
        std::cerr << "Failed to load GPU particle shader" << std::endl;
        // Don't fail init, just warn
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

    // Particle lifeRatio (instanced)
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (void*)offsetof(InstanceData, lifeRatio));
    glVertexAttribDivisor(5, 1);

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

void ParticleSystem::RenderFromGPU(const std::shared_ptr<ParticleEmitter>& emitter, Camera* camera) {
    if (!m_GPUShader) return;
    
    unsigned int ssbo = emitter->GetParticleSSBO();
    unsigned int activeCount = emitter->GetActiveParticleCount();
    
    if (ssbo == 0 || activeCount == 0) return;
    
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
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    
    // Draw using vertex pulling (no VBO needed for attributes, but we need a VAO bound)
    // We draw 6 vertices per particle (quad)
    glBindVertexArray(m_QuadVAO); // Reuse QuadVAO just to have something bound
    glDrawArrays(GL_TRIANGLES, 0, activeCount * 6);
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
    
    m_TrailShader->Use();
    m_TrailShader->SetMat4("view", camera->GetViewMatrix().m);
    m_TrailShader->SetMat4("projection", camera->GetProjectionMatrix().m);
    
    Vec3 cameraPos = camera->GetPosition();
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    
    for (const auto& emitter : m_Emitters) {
        if (!emitter->GetEnableTrails()) continue;
        
        // Collect all trail geometry for this emitter
        std::vector<float> vertices;
        std::vector<unsigned int> indices;
        
        GenerateTrailGeometry(emitter.get(), cameraPos, vertices, indices);
        
        if (vertices.empty() || indices.empty()) continue;
        
        // Upload geometry to GPU
        glBindVertexArray(m_TrailVAO);
        
        glBindBuffer(GL_ARRAY_BUFFER, m_TrailVBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_TrailEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_DYNAMIC_DRAW);
        
        // Setup vertex attributes
        // Position (3 floats)
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
        
        // TexCoord (2 floats)
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
        
        // Color (4 floats)
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(5 * sizeof(float)));
        
        // Bind texture if available
        auto trailTexture = emitter->GetTrailTexture();
        if (trailTexture) {
            trailTexture->Bind(0);
            m_TrailShader->SetInt("trailTexture", 0);
            m_TrailShader->SetInt("useTexture", 1);
        } else {
            m_TrailShader->SetInt("useTexture", 0);
        }
        
        // Draw trails
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);
    }
    
    glBindVertexArray(0);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

void ParticleSystem::GenerateTrailGeometry(const ParticleEmitter* emitter,
                                           const Vec3& cameraPos,
                                           std::vector<float>& vertices,
                                           std::vector<unsigned int>& indices) {
    const auto& particles = emitter->GetParticles();
    unsigned int vertexOffset = 0;
    
    for (const auto& particle : particles) {
        if (!particle.active || !particle.hasTrail || !particle.trail) continue;
        
        const auto& points = particle.trail->GetPoints();
        if (points.size() < 2) continue; // Need at least 2 points for a trail
        
        // Generate ribbon geometry
        for (size_t i = 0; i < points.size() - 1; ++i) {
            const TrailPoint& p1 = points[i];
            const TrailPoint& p2 = points[i + 1];
            
            // Calculate perpendicular vector for ribbon width
            Vec3 forward = p2.position - p1.position;
            float length = sqrtf(forward.x * forward.x + forward.y * forward.y + forward.z * forward.z);
            if (length < 0.001f) continue; // Skip degenerate segments
            
            forward.x /= length;
            forward.y /= length;
            forward.z /= length;
            
            // Get vector to camera for billboarding
            Vec3 toCamera = cameraPos - p1.position;
            
            // Cross product to get perpendicular (right) vector
            Vec3 right;
            right.x = forward.y * toCamera.z - forward.z * toCamera.y;
            right.y = forward.z * toCamera.x - forward.x * toCamera.z;
            right.z = forward.x * toCamera.y - forward.y * toCamera.x;
            
            float rightLength = sqrtf(right.x * right.x + right.y * right.y + right.z * right.z);
            if (rightLength < 0.001f) {
                // Fallback if vectors are parallel
                right = Vec3(0, 1, 0);
                rightLength = 1.0f;
            }
            
            right.x /= rightLength;
            right.y /= rightLength;
            right.z /= rightLength;
            
            // Calculate fade based on trail color mode
            Vec4 color1 = p1.color;
            Vec4 color2 = p2.color;
            
            if (emitter->GetTrailColorMode() == TrailColorMode::FadeToTransparent) {
                // Fade alpha based on age
                color1.w *= (1.0f - p1.lifeRatio);
                color2.w *= (1.0f - p2.lifeRatio);
            }
            
            // Create quad vertices (two triangles)
            float halfWidth1 = p1.width * 0.5f;
            float halfWidth2 = p2.width * 0.5f;
            
            // Vertex 1 (p1 - right)
            Vec3 v1 = p1.position - right * halfWidth1;
            vertices.push_back(v1.x);
            vertices.push_back(v1.y);
            vertices.push_back(v1.z);
            vertices.push_back(static_cast<float>(i) / (points.size() - 1)); // U coord
            vertices.push_back(0.0f); // V coord
            vertices.push_back(color1.x);
            vertices.push_back(color1.y);
            vertices.push_back(color1.z);
            vertices.push_back(color1.w);
            
            // Vertex 2 (p1 + right)
            Vec3 v2 = p1.position + right * halfWidth1;
            vertices.push_back(v2.x);
            vertices.push_back(v2.y);
            vertices.push_back(v2.z);
            vertices.push_back(static_cast<float>(i) / (points.size() - 1));
            vertices.push_back(1.0f);
            vertices.push_back(color1.x);
            vertices.push_back(color1.y);
            vertices.push_back(color1.z);
            vertices.push_back(color1.w);
            
            // Vertex 3 (p2 - right)
            Vec3 v3 = p2.position - right * halfWidth2;
            vertices.push_back(v3.x);
            vertices.push_back(v3.y);
            vertices.push_back(v3.z);
            vertices.push_back(static_cast<float>(i + 1) / (points.size() - 1));
            vertices.push_back(0.0f);
            vertices.push_back(color2.x);
            vertices.push_back(color2.y);
            vertices.push_back(color2.z);
            vertices.push_back(color2.w);
            
            // Vertex 4 (p2 + right)
            Vec3 v4 = p2.position + right * halfWidth2;
            vertices.push_back(v4.x);
            vertices.push_back(v4.y);
            vertices.push_back(v4.z);
            vertices.push_back(static_cast<float>(i + 1) / (points.size() - 1));
            vertices.push_back(1.0f);
            vertices.push_back(color2.x);
            vertices.push_back(color2.y);
            vertices.push_back(color2.z);
            vertices.push_back(color2.w);
            
            // Create indices for two triangles
            indices.push_back(vertexOffset + 0);
            indices.push_back(vertexOffset + 1);
            indices.push_back(vertexOffset + 2);
            
            indices.push_back(vertexOffset + 1);
            indices.push_back(vertexOffset + 3);
            indices.push_back(vertexOffset + 2);
            
            vertexOffset += 4;
        }
    }
}
