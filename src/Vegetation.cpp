#include "Vegetation.h"
#include "Shader.h"
#include "Texture.h"
#include "Terrain.h"
#include <iostream>
#include <algorithm>
#include <cmath>

Vegetation::Vegetation()
    : m_RNG(std::random_device{}())
{
}

Vegetation::~Vegetation() {
    if (m_VAO) glDeleteVertexArrays(1, &m_VAO);
    if (m_VBO) glDeleteBuffers(1, &m_VBO);
    if (m_InstanceVBO) glDeleteBuffers(1, &m_InstanceVBO);
}

bool Vegetation::Init() {
    CreateGrassMesh();
    
    // Create instance buffer
    glGenBuffers(1, &m_InstanceVBO);
    
    m_Initialized = true;
    std::cout << "Vegetation system initialized" << std::endl;
    return true;
}

void Vegetation::CreateGrassMesh() {
    // Create a grass blade mesh (triangle strip)
    // Each blade has multiple segments for better bending
    const int segments = 4;
    std::vector<float> vertices;
    
    for (int i = 0; i <= segments; ++i) {
        float t = static_cast<float>(i) / segments;
        float y = t * m_BladeHeight;
        float width = m_BladeWidth * (1.0f - t * 0.7f); // Taper toward tip
        
        // Left vertex
        vertices.push_back(-width * 0.5f);  // x
        vertices.push_back(y);               // y
        vertices.push_back(0.0f);            // z
        vertices.push_back(0.0f);            // u
        vertices.push_back(t);               // v
        
        // Right vertex
        vertices.push_back(width * 0.5f);   // x
        vertices.push_back(y);               // y
        vertices.push_back(0.0f);            // z
        vertices.push_back(1.0f);            // u
        vertices.push_back(t);               // v
    }
    
    m_VertexCount = static_cast<int>(vertices.size() / 5);
    
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);
    
    glBindVertexArray(m_VAO);
    
    // Grass mesh data
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    
    // Position (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // TexCoord (location 1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
}

void Vegetation::GenerateFromTerrain(Terrain* terrain, int splatChannel, float minWeight) {
    if (!terrain) return;
    
    Clear();
    
    float terrainWidth = terrain->GetWidth();
    float terrainDepth = terrain->GetDepth();
    Vec3 terrainPos = terrain->GetPosition();
    
    // Calculate number of grass instances based on density
    float area = terrainWidth * terrainDepth;
    int targetCount = static_cast<int>(area * m_Density);
    
    std::uniform_real_distribution<float> distX(0.0f, terrainWidth);
    std::uniform_real_distribution<float> distZ(0.0f, terrainDepth);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::uniform_real_distribution<float> distRot(0.0f, 6.28318f);
    std::uniform_real_distribution<float> distScale(0.7f, 1.3f);
    
    m_Instances.reserve(targetCount);
    
    for (int i = 0; i < targetCount * 2; ++i) { // Over-sample then filter
        float localX = distX(m_RNG);
        float localZ = distZ(m_RNG);
        
        float worldX = terrainPos.x + localX;
        float worldZ = terrainPos.z + localZ;
        
        // Get height from terrain
        float height = terrain->GetHeightAt(worldX, worldZ);
        
        // Get slope - don't spawn on steep slopes
        Vec3 normal = terrain->GetNormalAt(worldX, worldZ);
        if (normal.y < 0.7f) continue; // Skip steep slopes
        
        // TODO: Sample splatmap for spawn probability
        // For now, use random with height-based probability
        float spawnProb = 1.0f - (height / terrain->GetHeightScale()) * 0.5f;
        spawnProb = std::max(0.0f, std::min(1.0f, spawnProb));
        
        if (dist01(m_RNG) > spawnProb) continue;
        
        GrassInstance instance;
        instance.position = Vec3(worldX, height, worldZ);
        instance.rotation = distRot(m_RNG);
        instance.scale = distScale(m_RNG);
        instance.colorVar = dist01(m_RNG);
        instance.windPhase = dist01(m_RNG) * 6.28318f;
        
        m_Instances.push_back(instance);
        
        if (m_Instances.size() >= static_cast<size_t>(targetCount)) break;
    }
    
    m_BufferDirty = true;
    std::cout << "Generated " << m_Instances.size() << " grass instances" << std::endl;
}

void Vegetation::AddGrassRegion(const Vec3& center, float radius, int count, float baseHeight) {
    std::uniform_real_distribution<float> distAngle(0.0f, 6.28318f);
    std::uniform_real_distribution<float> distRadius(0.0f, radius);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::uniform_real_distribution<float> distRot(0.0f, 6.28318f);
    std::uniform_real_distribution<float> distScale(0.7f, 1.3f);
    
    for (int i = 0; i < count; ++i) {
        float angle = distAngle(m_RNG);
        float r = std::sqrt(distRadius(m_RNG) / radius) * radius; // Uniform distribution in circle
        
        GrassInstance instance;
        instance.position.x = center.x + std::cos(angle) * r;
        instance.position.y = baseHeight;
        instance.position.z = center.z + std::sin(angle) * r;
        instance.rotation = distRot(m_RNG);
        instance.scale = distScale(m_RNG);
        instance.colorVar = dist01(m_RNG);
        instance.windPhase = dist01(m_RNG) * 6.28318f;
        
        m_Instances.push_back(instance);
    }
    
    m_BufferDirty = true;
}

void Vegetation::UpdateInstances(const Vec3& cameraPos) {
    m_VisibleInstances.clear();
    
    float maxDistSq = m_MaxDistance * m_MaxDistance;
    
    for (const auto& instance : m_Instances) {
        float dx = instance.position.x - cameraPos.x;
        float dz = instance.position.z - cameraPos.z;
        float distSq = dx * dx + dz * dz;
        
        if (distSq < maxDistSq) {
            m_VisibleInstances.push_back(instance);
        }
    }
    
    m_VisibleCount = static_cast<int>(m_VisibleInstances.size());
    m_BufferDirty = true;
}

void Vegetation::UpdateInstanceBuffer() {
    if (!m_BufferDirty || m_VisibleInstances.empty()) return;
    
    // Pack instance data for GPU
    // Format: position.xyz, rotation, scale, colorVar, windPhase (7 floats per instance)
    std::vector<float> instanceData;
    instanceData.reserve(m_VisibleInstances.size() * 7);
    
    for (const auto& inst : m_VisibleInstances) {
        instanceData.push_back(inst.position.x);
        instanceData.push_back(inst.position.y);
        instanceData.push_back(inst.position.z);
        instanceData.push_back(inst.rotation);
        instanceData.push_back(inst.scale);
        instanceData.push_back(inst.colorVar);
        instanceData.push_back(inst.windPhase);
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, m_InstanceVBO);
    glBufferData(GL_ARRAY_BUFFER, instanceData.size() * sizeof(float), instanceData.data(), GL_DYNAMIC_DRAW);
    
    m_BufferDirty = false;
}

void Vegetation::Render(Shader* shader, const Mat4& view, const Mat4& projection, float time) {
    if (!m_Initialized || !shader || m_VisibleInstances.empty()) return;
    
    UpdateInstanceBuffer();
    
    shader->Use();
    shader->SetMat4("u_View", view.m);
    shader->SetMat4("u_Projection", projection.m);
    shader->SetFloat("u_Time", time);
    shader->SetFloat("u_WindStrength", m_WindStrength);
    shader->SetFloat("u_WindSpeed", m_WindSpeed);
    shader->SetVec2("u_WindDirection", m_WindDirection.x, m_WindDirection.y);
    shader->SetFloat("u_BladeHeight", m_BladeHeight);
    shader->SetFloat("u_FadeStart", m_FadeStart);
    shader->SetFloat("u_FadeEnd", m_MaxDistance);
    shader->SetVec3("u_ColorBase", m_ColorBase.x, m_ColorBase.y, m_ColorBase.z);
    shader->SetVec3("u_ColorTip", m_ColorTip.x, m_ColorTip.y, m_ColorTip.z);
    
    // Bind textures
    if (m_GrassTexture) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_GrassTexture->GetID());
        shader->SetInt("u_GrassTexture", 0);
    }
    
    if (m_NoiseTexture) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, m_NoiseTexture->GetID());
        shader->SetInt("u_NoiseTexture", 1);
    }
    
    // Bind VAO and setup instanced attributes
    glBindVertexArray(m_VAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, m_InstanceVBO);
    
    // Instance position (location 2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);
    
    // Instance rotation (location 3)
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(3);
    glVertexAttribDivisor(3, 1);
    
    // Instance scale (location 4)
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(4 * sizeof(float)));
    glEnableVertexAttribArray(4);
    glVertexAttribDivisor(4, 1);
    
    // Instance colorVar (location 5)
    glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(5);
    glVertexAttribDivisor(5, 1);
    
    // Instance windPhase (location 6)
    glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(6);
    glVertexAttribDivisor(6, 1);
    
    // Enable alpha blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);
    
    // Draw instanced
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, m_VertexCount, m_VisibleCount);
    
    // Cleanup
    glEnable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    
    // Reset divisors
    glVertexAttribDivisor(2, 0);
    glVertexAttribDivisor(3, 0);
    glVertexAttribDivisor(4, 0);
    glVertexAttribDivisor(5, 0);
    glVertexAttribDivisor(6, 0);
    
    glBindVertexArray(0);
}

void Vegetation::Clear() {
    m_Instances.clear();
    m_VisibleInstances.clear();
    m_VisibleCount = 0;
    m_BufferDirty = true;
}
