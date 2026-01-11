#include "FluidEmitter.h"
#include <cstdlib>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

FluidEmitter::FluidEmitter()
    : m_Position(0, 0, 0)
    , m_Velocity(0, -1, 0)
    , m_EmissionRate(100.0f)
    , m_FluidType(0)
    , m_Enabled(true)
    , m_EmissionShape(EmissionShape::Point)
    , m_BoxExtents(0.1f, 0.1f, 0.1f)
    , m_SphereRadius(0.1f)
    , m_DiscRadius(0.1f)
    , m_DiscNormal(0, 1, 0)
    , m_VelocitySpread(0.1f)
    , m_ParticleLifetime(-1.0f)
    , m_AccumulatedTime(0.0f)
    , m_TotalEmitted(0)
{
}

FluidEmitter::~FluidEmitter() {
}

void FluidEmitter::Update(float deltaTime, std::vector<FluidParticle>& outParticles) {
    if (!m_Enabled || m_EmissionRate <= 0.0f) {
        return;
    }
    
    m_AccumulatedTime += deltaTime;
    
    // Calculate how many particles to emit this frame
    float particlesPerFrame = m_EmissionRate * deltaTime;
    int particlesToEmit = static_cast<int>(particlesPerFrame);
    
    // Handle fractional particles with accumulation
    float emissionInterval = 1.0f / m_EmissionRate;
    while (m_AccumulatedTime >= emissionInterval) {
        m_AccumulatedTime -= emissionInterval;
        particlesToEmit++;
    }
    
    // Emit particles
    for (int i = 0; i < particlesToEmit; ++i) {
        FluidParticle particle;
        particle.position = GetRandomPositionInShape();
        particle.velocity = GetRandomVelocity();
        particle.fluidType = m_FluidType;
        particle.lifetime = m_ParticleLifetime;
        particle.age = 0.0f;
        particle.active = true;
        
        outParticles.push_back(particle);
        m_TotalEmitted++;
    }
}

Vec3 FluidEmitter::GetRandomPositionInShape() const {
    switch (m_EmissionShape) {
        case EmissionShape::Point:
            return m_Position;
            
        case EmissionShape::Box: {
            Vec3 offset(
                RandomFloat(-m_BoxExtents.x, m_BoxExtents.x),
                RandomFloat(-m_BoxExtents.y, m_BoxExtents.y),
                RandomFloat(-m_BoxExtents.z, m_BoxExtents.z)
            );
            return m_Position + offset;
        }
        
        case EmissionShape::Sphere: {
            // Random point in sphere using rejection sampling
            Vec3 offset;
            do {
                offset = Vec3(
                    RandomFloat(-1.0f, 1.0f),
                    RandomFloat(-1.0f, 1.0f),
                    RandomFloat(-1.0f, 1.0f)
                );
            } while (offset.LengthSquared() > 1.0f);
            
            return m_Position + offset * m_SphereRadius;
        }
        
        case EmissionShape::Disc: {
            // Random point on disc
            float angle = RandomFloat(0.0f, 2.0f * static_cast<float>(M_PI));
            float radius = std::sqrt(RandomFloat(0.0f, 1.0f)) * m_DiscRadius;
            
            // Create orthogonal basis from disc normal
            Vec3 tangent = std::abs(m_DiscNormal.y) < 0.9f ? 
                          Vec3(0, 1, 0) : Vec3(1, 0, 0);
            Vec3 bitangent = m_DiscNormal.Cross(tangent).Normalized();
            tangent = bitangent.Cross(m_DiscNormal).Normalized();
            
            Vec3 offset = tangent * (std::cos(angle) * radius) + 
                         bitangent * (std::sin(angle) * radius);
            
            return m_Position + offset;
        }
        
        case EmissionShape::Mesh:
            // TODO: Implement mesh surface emission
            return m_Position;
            
        default:
            return m_Position;
    }
}

Vec3 FluidEmitter::GetRandomVelocity() const {
    Vec3 baseVelocity = m_Velocity;
    
    if (m_VelocitySpread > 0.0f) {
        Vec3 randomOffset(
            RandomFloat(-m_VelocitySpread, m_VelocitySpread),
            RandomFloat(-m_VelocitySpread, m_VelocitySpread),
            RandomFloat(-m_VelocitySpread, m_VelocitySpread)
        );
        baseVelocity = baseVelocity + randomOffset;
    }
    
    return baseVelocity;
}

float FluidEmitter::RandomFloat(float min, float max) const {
    float random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return min + random * (max - min);
}
