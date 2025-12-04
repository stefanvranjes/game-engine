#pragma once

#include "Particle.h"
#include "Texture.h"
#include "Math/Vec3.h"
#include "Math/Vec4.h"
#include <vector>
#include <memory>

enum class EmitterShape {
    Point,
    Sphere,
    Cone,
    Box
};

enum class BlendMode {
    Additive,
    Alpha
};

class ParticleEmitter {
public:
    ParticleEmitter(const Vec3& position, int maxParticles = 1000);
    ~ParticleEmitter();

    void Update(float deltaTime);
    const std::vector<Particle>& GetParticles() const { return m_Particles; }
    
    // Configuration
    void SetPosition(const Vec3& pos) { m_Position = pos; }
    Vec3 GetPosition() const { return m_Position; }
    
    void SetSpawnRate(float rate) { m_SpawnRate = rate; }
    void SetParticleLifetime(float lifetime) { m_ParticleLifetime = lifetime; }
    void SetVelocityRange(const Vec3& min, const Vec3& max) { m_VelocityMin = min; m_VelocityMax = max; }
    void SetColorRange(const Vec4& start, const Vec4& end) { m_ColorStart = start; m_ColorEnd = end; }
    void SetSizeRange(float start, float end) { m_SizeStart = start; m_SizeEnd = end; }
    void SetGravity(const Vec3& gravity) { m_Gravity = gravity; }
    void SetEmitterShape(EmitterShape shape) { m_Shape = shape; }
    void SetConeAngle(float angle) { m_ConeAngle = angle; }
    void SetSphereRadius(float radius) { m_SphereRadius = radius; }
    void SetBoxSize(const Vec3& size) { m_BoxSize = size; }
    void SetBlendMode(BlendMode mode) { m_BlendMode = mode; }
    void SetTexture(std::shared_ptr<Texture> texture) { m_Texture = texture; }
    void SetParent(class GameObject* parent) { m_Parent = parent; }
    void SetAtlasSize(int rows, int cols) { m_AtlasRows = rows; m_AtlasCols = cols; }
    void SetAnimationSpeed(float speed) { m_AnimationSpeed = speed; }
    void SetLoopAnimation(bool loop) { m_LoopAnimation = loop; }
    
    BlendMode GetBlendMode() const { return m_BlendMode; }
    std::shared_ptr<Texture> GetTexture() const { return m_Texture; }
    GameObject* GetParent() const { return m_Parent; }
    int GetAtlasRows() const { return m_AtlasRows; }
    int GetAtlasCols() const { return m_AtlasCols; }
    float GetAnimationSpeed() const { return m_AnimationSpeed; }
    bool IsLoopAnimation() const { return m_LoopAnimation; }
    bool IsActive() const { return m_Active; }
    void SetActive(bool active) { m_Active = active; }
    
    // Presets
    static std::shared_ptr<ParticleEmitter> CreateFire(const Vec3& position);
    static std::shared_ptr<ParticleEmitter> CreateSmoke(const Vec3& position);
    static std::shared_ptr<ParticleEmitter> CreateSparks(const Vec3& position);
    static std::shared_ptr<ParticleEmitter> CreateMagic(const Vec3& position);

private:
    void SpawnParticle();
    Vec3 GetRandomVelocity();
    float RandomFloat(float min, float max);
    
    std::vector<Particle> m_Particles;
    int m_MaxParticles;
    
    // Emitter properties
    Vec3 m_Position;
    float m_SpawnRate;           // Particles per second
    float m_ParticleLifetime;
    Vec3 m_VelocityMin;
    Vec3 m_VelocityMax;
    Vec4 m_ColorStart;
    Vec4 m_ColorEnd;
    float m_SizeStart;
    float m_SizeEnd;
    Vec3 m_Gravity;
    EmitterShape m_Shape;
    float m_ConeAngle;
    float m_SphereRadius;
    Vec3 m_BoxSize;
    BlendMode m_BlendMode;
    std::shared_ptr<Texture> m_Texture;
    GameObject* m_Parent;
    bool m_Active;
    
    // Internal state
    float m_SpawnAccumulator;
    
    // Atlas animation
    int m_AtlasRows;
    int m_AtlasCols;
    float m_AnimationSpeed;
    bool m_LoopAnimation;
};
