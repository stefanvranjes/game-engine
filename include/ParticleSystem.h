#pragma once

#include "ParticleEmitter.h"
#include "Shader.h"
#include "Camera.h"
#include <vector>
#include <memory>

class ParticleSystem {
public:
    ParticleSystem();
    ~ParticleSystem();

    bool Init();
    void Update(float deltaTime);
    void Render(Camera* camera, class GBuffer* gbuffer = nullptr);
    void Shutdown();

    void AddEmitter(std::shared_ptr<ParticleEmitter> emitter);
    void RemoveEmitter(std::shared_ptr<ParticleEmitter> emitter);
    void ClearEmitters();
    
    std::vector<std::shared_ptr<ParticleEmitter>>& GetEmitters() { return m_Emitters; }

private:
    void SetupQuadMesh();
    void UpdateInstanceBuffer(const std::vector<Particle>& particles);
    
    std::vector<std::shared_ptr<ParticleEmitter>> m_Emitters;
    std::unique_ptr<Shader> m_Shader;
    
    // GPU resources
    unsigned int m_QuadVAO;
    unsigned int m_QuadVBO;
    unsigned int m_InstanceVBO;
    
    // Instance data buffer
    struct InstanceData {
        float position[3];
        float color[4];
        float size;
    };
    std::vector<InstanceData> m_InstanceData;
};
