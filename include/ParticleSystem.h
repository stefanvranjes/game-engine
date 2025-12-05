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
    
    // Global Budget
    void SetGlobalParticleLimit(int limit) { m_GlobalParticleLimit = limit; }
    int GetGlobalParticleLimit() const { return m_GlobalParticleLimit; }
    int GetTotalActiveParticles() const { return m_TotalActiveParticles; }

    // Physics Enhancements
    void SetGlobalWind(const Vec3& wind) { m_GlobalWind = wind; }
    Vec3 GetGlobalWind() const { return m_GlobalWind; }
    
    struct Attractor {
        Vec3 position;
        float strength; // Positive = Pull, Negative = Push
    };
    void AddAttractor(const Attractor& attractor) { m_Attractors.push_back(attractor); }
    void ClearAttractors() { m_Attractors.clear(); }
    std::vector<Attractor>& GetAttractors() { return m_Attractors; }

private:
    std::vector<std::shared_ptr<ParticleEmitter>> m_Emitters;
    int m_GlobalParticleLimit = 100000;
    int m_TotalActiveParticles = 0;
    
    // Physics
    Vec3 m_GlobalWind = Vec3(0,0,0);
    std::vector<Attractor> m_Attractors;
    
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
