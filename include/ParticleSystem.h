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
    void Update(float deltaTime, const Vec3& cameraPos = Vec3(0,0,0));
    void Render(Camera* camera, class GBuffer* gbuffer = nullptr);
    void Shutdown();

    void AddEmitter(std::shared_ptr<ParticleEmitter> emitter);
    void RemoveEmitter(std::shared_ptr<ParticleEmitter> emitter);
    void ClearEmitters();
    const std::vector<std::shared_ptr<ParticleEmitter>>& GetEmitters() const { return m_Emitters; }
    struct InstanceData {
        float position[3];
        float color[4];
        float size;
        float lifeRatio;
    };
    std::vector<InstanceData> m_InstanceData;

private:
    std::vector<std::shared_ptr<ParticleEmitter>> m_Emitters;
    
    // Shaders
    std::unique_ptr<Shader> m_Shader;
    std::unique_ptr<Shader> m_GPUShader;
    std::unique_ptr<Shader> m_SortInitShader;
    std::unique_ptr<Shader> m_SortStepShader;
    std::unique_ptr<Shader> m_TrailShader;
    
    // Buffers
    unsigned int m_QuadVAO = 0;
    unsigned int m_QuadVBO = 0;
    unsigned int m_InstanceVBO = 0;
    
    // Trail buffers (Legacy/CPU)
    unsigned int m_TrailVAO = 0;
    unsigned int m_TrailVBO = 0;
    unsigned int m_TrailEBO = 0;
    
    // Helper methods
    void SetupQuadMesh();
    void RenderTrails(Camera* camera);
    void RenderFromGPU(const std::shared_ptr<ParticleEmitter>& emitter, Camera* camera);
    void SortParticlesGPU(const std::shared_ptr<ParticleEmitter>& emitter, Camera* camera);
    void RenderTrailsGPU(const std::shared_ptr<ParticleEmitter>& emitter, Camera* camera);
};
