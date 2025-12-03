#include "ParticleEmitter.h"
#include <cstdlib>
#include <ctime>
#include <cmath>

ParticleEmitter::ParticleEmitter(const Vec3& position, int maxParticles)
    : m_Position(position)
    , m_MaxParticles(maxParticles)
    , m_SpawnRate(50.0f)
    , m_ParticleLifetime(2.0f)
    , m_VelocityMin(-1, 0, -1)
    , m_VelocityMax(1, 3, 1)
    , m_ColorStart(1, 1, 1, 1)
    , m_ColorEnd(1, 1, 1, 0)
    , m_SizeStart(1.0f)
    , m_SizeEnd(0.5f)
    , m_Gravity(0, -2.0f, 0)
    , m_Shape(EmitterShape::Point)
    , m_ConeAngle(45.0f)
    , m_SphereRadius(1.0f)
    , m_BoxSize(1, 1, 1)
    , m_BlendMode(BlendMode::Alpha)
    , m_Texture(nullptr)
    , m_Parent(nullptr)
    , m_Active(true)
    , m_SpawnAccumulator(0.0f)
{
    m_Particles.resize(maxParticles);
    static bool seeded = false;
    if (!seeded) {
        srand(static_cast<unsigned int>(time(nullptr)));
        seeded = true;
    }
}

ParticleEmitter::~ParticleEmitter() {
}

void ParticleEmitter::Update(float deltaTime) {
    if (!m_Active) return;
    
    // Update position from parent if attached
    if (m_Parent) {
        m_Position = m_Parent->GetWorldTransform().GetTranslation();
    }
    
    // Spawn new particles
    m_SpawnAccumulator += deltaTime * m_SpawnRate;
    int particlesToSpawn = static_cast<int>(m_SpawnAccumulator);
    m_SpawnAccumulator -= particlesToSpawn;
    
    for (int i = 0; i < particlesToSpawn; ++i) {
        SpawnParticle();
    }
    
    // Update existing particles
    for (auto& particle : m_Particles) {
        if (!particle.active) continue;
        
        particle.age += deltaTime;
        
        // Kill old particles
        if (particle.age >= particle.lifetime) {
            particle.active = false;
            continue;
        }
        
        // Update physics
        particle.velocity = particle.velocity + m_Gravity * deltaTime;
        particle.position = particle.position + particle.velocity * deltaTime;
        
        // Interpolate color and size based on age
        float t = particle.age / particle.lifetime;
        particle.color = Vec4(
            m_ColorStart.x + (m_ColorEnd.x - m_ColorStart.x) * t,
            m_ColorStart.y + (m_ColorEnd.y - m_ColorStart.y) * t,
            m_ColorStart.z + (m_ColorEnd.z - m_ColorStart.z) * t,
            m_ColorStart.w + (m_ColorEnd.w - m_ColorStart.w) * t
        );
        particle.size = m_SizeStart + (m_SizeEnd - m_SizeStart) * t;
    }
}

void ParticleEmitter::SpawnParticle() {
    // Find inactive particle
    for (auto& particle : m_Particles) {
        if (!particle.active) {
            particle.active = true;
            particle.age = 0.0f;
            particle.lifetime = m_ParticleLifetime * (0.8f + RandomFloat(0, 0.4f)); // Slight variation
            
            // Set position based on emitter shape
            switch (m_Shape) {
                case EmitterShape::Point:
                    particle.position = m_Position;
                    break;
                case EmitterShape::Sphere: {
                    float theta = RandomFloat(0, 2.0f * 3.14159f);
                    float phi = RandomFloat(0, 3.14159f);
                    float r = RandomFloat(0, m_SphereRadius);
                    particle.position = m_Position + Vec3(
                        r * sin(phi) * cos(theta),
                        r * sin(phi) * sin(theta),
                        r * cos(phi)
                    );
                    break;
                }
                case EmitterShape::Box:
                    particle.position = m_Position + Vec3(
                        RandomFloat(-m_BoxSize.x, m_BoxSize.x),
                        RandomFloat(-m_BoxSize.y, m_BoxSize.y),
                        RandomFloat(-m_BoxSize.z, m_BoxSize.z)
                    );
                    break;
                case EmitterShape::Cone:
                default:
                    particle.position = m_Position;
                    break;
            }
            
            particle.velocity = GetRandomVelocity();
            particle.color = m_ColorStart;
            particle.size = m_SizeStart;
            return;
        }
    }
}

Vec3 ParticleEmitter::GetRandomVelocity() {
    return Vec3(
        RandomFloat(m_VelocityMin.x, m_VelocityMax.x),
        RandomFloat(m_VelocityMin.y, m_VelocityMax.y),
        RandomFloat(m_VelocityMin.z, m_VelocityMax.z)
    );
}

float ParticleEmitter::RandomFloat(float min, float max) {
    return min + (max - min) * (static_cast<float>(rand()) / RAND_MAX);
}

// Preset factory methods
std::shared_ptr<ParticleEmitter> ParticleEmitter::CreateFire(const Vec3& position) {
    auto emitter = std::make_shared<ParticleEmitter>(position, 500);
    emitter->SetSpawnRate(100.0f);
    emitter->SetParticleLifetime(1.5f);
    emitter->SetVelocityRange(Vec3(-0.5f, 1.0f, -0.5f), Vec3(0.5f, 3.0f, 0.5f));
    emitter->SetColorRange(Vec4(1.0f, 1.0f, 0.2f, 1.0f), Vec4(1.0f, 0.2f, 0.0f, 0.0f)); // Yellow to red to transparent
    emitter->SetSizeRange(0.3f, 1.0f);
    emitter->SetGravity(Vec3(0, 0.5f, 0)); // Slight upward drift
    emitter->SetBlendMode(BlendMode::Additive);
    return emitter;
}

std::shared_ptr<ParticleEmitter> ParticleEmitter::CreateSmoke(const Vec3& position) {
    auto emitter = std::make_shared<ParticleEmitter>(position, 300);
    emitter->SetSpawnRate(30.0f);
    emitter->SetParticleLifetime(4.0f);
    emitter->SetVelocityRange(Vec3(-0.3f, 0.5f, -0.3f), Vec3(0.3f, 1.5f, 0.3f));
    emitter->SetColorRange(Vec4(0.8f, 0.8f, 0.8f, 0.8f), Vec4(0.5f, 0.5f, 0.5f, 0.0f)); // Gray to transparent
    emitter->SetSizeRange(0.5f, 2.0f);
    emitter->SetGravity(Vec3(0, 0.2f, 0)); // Slow upward drift
    emitter->SetBlendMode(BlendMode::Alpha);
    return emitter;
}

std::shared_ptr<ParticleEmitter> ParticleEmitter::CreateSparks(const Vec3& position) {
    auto emitter = std::make_shared<ParticleEmitter>(position, 200);
    emitter->SetSpawnRate(150.0f);
    emitter->SetParticleLifetime(0.8f);
    emitter->SetVelocityRange(Vec3(-3.0f, 1.0f, -3.0f), Vec3(3.0f, 5.0f, 3.0f));
    emitter->SetColorRange(Vec4(1.0f, 1.0f, 0.5f, 1.0f), Vec4(1.0f, 0.5f, 0.0f, 0.0f)); // Bright yellow to orange
    emitter->SetSizeRange(0.1f, 0.1f); // Constant small size
    emitter->SetGravity(Vec3(0, -9.8f, 0)); // Strong gravity
    emitter->SetBlendMode(BlendMode::Additive);
    return emitter;
}

std::shared_ptr<ParticleEmitter> ParticleEmitter::CreateMagic(const Vec3& position) {
    auto emitter = std::make_shared<ParticleEmitter>(position, 400);
    emitter->SetSpawnRate(80.0f);
    emitter->SetParticleLifetime(2.5f);
    emitter->SetVelocityRange(Vec3(-1.0f, -1.0f, -1.0f), Vec3(1.0f, 1.0f, 1.0f));
    emitter->SetColorRange(Vec4(0.5f, 0.2f, 1.0f, 1.0f), Vec4(0.2f, 0.8f, 1.0f, 0.0f)); // Purple to cyan
    emitter->SetSizeRange(0.4f, 0.8f);
    emitter->SetGravity(Vec3(0, 0, 0)); // No gravity
    emitter->SetBlendMode(BlendMode::Additive);
    emitter->SetEmitterShape(EmitterShape::Sphere);
    emitter->SetSphereRadius(0.5f);
    return emitter;
}
