#include "ParticleEmitter.h"
#include "GameObject.h"
#include "NoiseGenerator.h"
#include "Shader.h"
#include "GLExtensions.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>

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
    , m_AtlasRows(1)
    , m_AtlasCols(1)
    , m_AnimationSpeed(1.0f)
    , m_LoopAnimation(true)
    , m_TrailTurbulence(0.0f)
    , m_TrailTurbulenceFrequency(1.0f)
    , m_TrailTurbulenceSpeed(1.0f)
    , m_Time(0.0f)
    , m_UseGPUCompute(false)
    , m_GPUPersistent(false)
    , m_ParticleSSBO(0)
    , m_AtomicCounterBuffer(0)
    , m_ActiveParticleCount(0)
    , m_ComputeShader(nullptr)
{
    m_Particles.resize(maxParticles);
    static bool seeded = false;
    if (!seeded) {
        srand(static_cast<unsigned int>(time(nullptr)));
        seeded = true;
    }
}

// ... (keep existing methods until GPU Compute implementation) ...

// GPU Compute implementation
void ParticleEmitter::SetUseGPUCompute(bool enable) {
    if (enable == m_UseGPUCompute) return;
    
    if (enable) {
        if (IsGPUComputeAvailable()) {
            InitGPUCompute();
            m_UseGPUCompute = true;
        } else {
            std::cerr << "GPU Compute not available, falling back to CPU" << std::endl;
            m_UseGPUCompute = false;
        }
    } else {
        ShutdownGPUCompute();
        m_UseGPUCompute = false;
    }
}

void ParticleEmitter::SetGPUPersistent(bool enable) {
    m_GPUPersistent = enable;
    if (m_GPUPersistent && !m_UseGPUCompute) {
        SetUseGPUCompute(true);
    }
}

bool ParticleEmitter::IsGPUComputeAvailable() const {
    return glDispatchCompute != nullptr && glMemoryBarrier != nullptr && glBindBufferBase != nullptr;
}

void ParticleEmitter::InitGPUCompute() {
    // Create compute shader
    m_ComputeShader = std::make_unique<Shader>();
    if (!m_ComputeShader->LoadComputeShader("shaders/particle_physics.comp")) {
        std::cerr << "Failed to load particle physics compute shader" << std::endl;
        m_ComputeShader = nullptr;
        return;
    }
    
    // Create SSBO for particle data
    glGenBuffers(1, &m_ParticleSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ParticleSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, m_Particles.size() * sizeof(Particle), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    
    // Create atomic counter buffer
    glGenBuffers(1, &m_AtomicCounterBuffer);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_AtomicCounterBuffer);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
    
    std::cout << "GPU Compute initialized for particle system" << std::endl;
}

void ParticleEmitter::ShutdownGPUCompute() {
    if (m_ParticleSSBO != 0) {
        glDeleteBuffers(1, &m_ParticleSSBO);
        m_ParticleSSBO = 0;
    }
    if (m_AtomicCounterBuffer != 0) {
        glDeleteBuffers(1, &m_AtomicCounterBuffer);
        m_AtomicCounterBuffer = 0;
    }
    m_ComputeShader = nullptr;
}

void ParticleEmitter::UpdateGPU(float deltaTime) {
    if (!m_ComputeShader || m_ParticleSSBO == 0) return;
    
    // Upload particle data to GPU if NOT persistent (or first run)
    // For persistent mode, we assume data is already there or initialized
    if (!m_GPUPersistent) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ParticleSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, m_Particles.size() * sizeof(Particle), m_Particles.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    } else {
        // If persistent, we might need to upload initial state once, but for now assume 
        // the compute shader handles respawning. 
        // However, if we just switched to persistent, we should upload current state.
        // For simplicity, we'll just upload if we have active particles on CPU side 
        // that we want to transfer, but typically persistent means "simulated entirely on GPU".
        // We'll skip upload here to avoid bandwidth.
    }
    
    // Reset atomic counter
    unsigned int zero = 0;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_AtomicCounterBuffer);
    glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(unsigned int), &zero);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
    
    // Bind buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ParticleSSBO);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 1, m_AtomicCounterBuffer);
    
    // Use compute shader
    m_ComputeShader->Use();
    
    // Set uniforms
    m_ComputeShader->SetFloat("deltaTime", deltaTime);
    m_ComputeShader->SetVec3("gravity", m_Gravity.x, m_Gravity.y, m_Gravity.z);
    m_ComputeShader->SetVec3("emitterPosition", m_Position.x, m_Position.y, m_Position.z);
    m_ComputeShader->SetFloat("particleLifetime", m_ParticleLifetime);
    m_ComputeShader->SetVec3("velocityMin", m_VelocityMin.x, m_VelocityMin.y, m_VelocityMin.z);
    m_ComputeShader->SetVec3("velocityMax", m_VelocityMax.x, m_VelocityMax.y, m_VelocityMax.z);
    m_ComputeShader->SetVec4("colorStart", m_ColorStart.x, m_ColorStart.y, m_ColorStart.z, m_ColorStart.w);
    m_ComputeShader->SetVec4("colorEnd", m_ColorEnd.x, m_ColorEnd.y, m_ColorEnd.z, m_ColorEnd.w);
    m_ComputeShader->SetFloat("sizeStart", m_SizeStart);
    m_ComputeShader->SetFloat("sizeEnd", m_SizeEnd);
    m_ComputeShader->SetInt("maxParticles", m_MaxParticles);
    m_ComputeShader->SetFloat("spawnRate", m_SpawnRate);
    m_ComputeShader->SetFloat("time", m_Time);
    
    // Dispatch compute shader
    unsigned int numWorkGroups = (m_MaxParticles + 255) / 256;
    m_ComputeShader->Dispatch(numWorkGroups, 1, 1);
    
    // Memory barrier
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);
    
    // If persistent, read back active count for rendering
    if (m_GPUPersistent) {
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_AtomicCounterBuffer);
        glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(unsigned int), &m_ActiveParticleCount);
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
    } else {
        // Download particle data from GPU (slow path)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ParticleSSBO);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, m_Particles.size() * sizeof(Particle), m_Particles.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
}

void ParticleEmitter::UpdateCPU(float deltaTime) {
    // Original CPU update code
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
        particle.lifeRatio = t;
        particle.color = Vec4(
            m_ColorStart.x + (m_ColorEnd.x - m_ColorStart.x) * t,
            m_ColorStart.y + (m_ColorEnd.y - m_ColorStart.y) * t,
            m_ColorStart.z + (m_ColorEnd.z - m_ColorStart.z) * t,
            m_ColorStart.w + (m_ColorEnd.w - m_ColorStart.w) * t
        );
        particle.size = m_SizeStart + (m_SizeEnd - m_SizeStart) * t;
        
        // Update trails
        if (particle.hasTrail && particle.trail) {
            // Calculate trail position with optional turbulence
            Vec3 trailPosition = particle.position;
            
            if (m_TrailTurbulence > 0.0f) {
                // Apply 3D noise for turbulence
                float noiseX = NoiseGenerator::Noise3D(
                    particle.position.x * m_TrailTurbulenceFrequency,
                    particle.position.y * m_TrailTurbulenceFrequency,
                    m_Time * m_TrailTurbulenceSpeed
                );
                float noiseY = NoiseGenerator::Noise3D(
                    particle.position.y * m_TrailTurbulenceFrequency,
                    particle.position.z * m_TrailTurbulenceFrequency,
                    m_Time * m_TrailTurbulenceSpeed + 100.0f
                );
                float noiseZ = NoiseGenerator::Noise3D(
                    particle.position.z * m_TrailTurbulenceFrequency,
                    particle.position.x * m_TrailTurbulenceFrequency,
                    m_Time * m_TrailTurbulenceSpeed + 200.0f
                );
                
                Vec3 noiseOffset(noiseX, noiseY, noiseZ);
                noiseOffset = noiseOffset * m_TrailTurbulence;
                trailPosition = trailPosition + noiseOffset;
            }
            
            // Add point to trail
            particle.trail->AddPoint(
                trailPosition,
                particle.color,
                m_TrailWidth * particle.size
            );
            
            // Update trail points
            particle.trail->Update(deltaTime);
        }
    }
}

