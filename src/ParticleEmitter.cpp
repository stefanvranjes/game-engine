#include "ParticleEmitter.h"
#include "GameObject.h"
#include "NoiseGenerator.h"
#include "Shader.h"
#include "GLExtensions.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <algorithm>

// Helper for bitonic sort size
static unsigned int NextPowerOfTwo(unsigned int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

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
    , m_EnableTurbulence(false)
    , m_TurbulenceStrength(1.0f)
    , m_SpawnAccumulator(0.0f)
    , m_ActiveParticleCount(0)
    , m_SpringSSBO(0)
    , m_ForceSSBO(0)
    , m_SortBufferSize(0)
    , m_IndirectBuffer(0)
    , m_ComputeShader(nullptr)
    , m_GridSSBO(0)
    , m_CollisionGridSize(1000000) // 1 Million cells
    , m_CollisionCellSize(2.0f)    // Cell size
    , m_PhysicsMode(PhysicsMode::Simple)
    , m_RestDensity(1000.0f)
    , m_GasConstant(2000.0f)
    , m_Viscosity(0.1f)
    , m_SmoothingRadius(2.0f)
    , m_GasConstant(2000.0f)
    , m_Viscosity(0.1f)
    , m_SmoothingRadius(2.0f)
    , m_Priority(5) // Default Medium Priority
    , m_VelocityInheritance(0.0f)
{
    m_Particles.resize(maxParticles);
    m_GPUParticles.resize(maxParticles); // Ensure GPU particles are also resized initially
    m_GPUParticles.resize(maxParticles); // Ensure GPU particles are also resized initially
    m_UseLOD = false;
    m_CurrentLODLevel = -1;
    m_CurrentDistance = 0.0f;
    
    static bool seeded = false;
    if (!seeded) {
        srand(static_cast<unsigned int>(time(nullptr)));
        seeded = true;
    }
}

void ParticleEmitter::UpdateLOD(float distance) {
    m_CurrentDistance = distance;
    
    if (m_LODs.empty()) {
        m_ActiveLOD = EmitterLOD();
        m_CurrentLODLevel = -1;
        return;
    }
    
    // Find appropriate LOD level
    int bestLevel = -1;
    for (size_t i = 0; i < m_LODs.size(); ++i) {
        if (distance >= m_LODs[i].distance) {
            bestLevel = static_cast<int>(i);
        } else {
            break; // Sorted by distance usually
        }
    }
    
    m_CurrentLODLevel = bestLevel;
    
    if (bestLevel != -1) {
        m_ActiveLOD = m_LODs[bestLevel];
    } else {
        // Closer than first LOD -> Full detail
        m_ActiveLOD = EmitterLOD();
    }
}

void ParticleEmitter::Update(float deltaTime, const Vec3& cameraPos) {
    if (!m_Active) return;

    if (m_UseLOD) {
        Vec3 diff = m_Position - cameraPos;
        float dist = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        UpdateLOD(dist);
    } else {
        m_ActiveLOD = EmitterLOD();
    }
    
    // Calculate Emitter Velocity for Inheritance
    if (deltaTime > 0.0001f) {
        m_CalculatedEmitterVelocity = (m_Position - m_LastPosition) / deltaTime;
    } else {
        m_CalculatedEmitterVelocity = Vec3(0,0,0);
    }
    
    // Animate turbulence time
    m_Time += deltaTime;

    if (m_UseGPUCompute) {
        UpdateGPU(deltaTime);
    } else {
        // CPU Update
        float effectiveSpawnRate = m_SpawnRate * m_ActiveLOD.emissionMultiplier;
        
        m_SpawnAccumulator += deltaTime * effectiveSpawnRate;
        while (m_SpawnAccumulator > 1.0f) {
            SpawnParticle();
            m_SpawnAccumulator -= 1.0f;
        }
        
        UpdateCPU(deltaTime);
        
        // Collisions
        if (m_ActiveLOD.enableCollisions) {
            if (m_EnableParticleCollisions) {
                HandleParticleCollisions(deltaTime);
            }
            
            // Shape Collisions
            for (auto& particle : m_Particles) {
                if (particle.active) HandleShapeCollisions(particle, deltaTime);
            }
        }
        
        // Update Last Position for next frame velocity calc
        m_LastPosition = m_Position;
    }
}
}

void ParticleEmitter::Burst(int count) {
    if (!m_Active) return;
    for (int i = 0; i < count; ++i) {
        SpawnParticle();
    }
}

void ParticleEmitter::SpawnParticle() {
    // Find inactive particle
    for (auto& p : m_Particles) {
        if (!p.active) {
            p.active = true;
            p.position = m_Position;
            
            // Randomize position based on shape
            if (m_Shape == EmitterShape::Sphere) {
                // Random point in sphere
                float theta = RandomFloat(0, 6.283185f);
                float phi = RandomFloat(0, 3.14159f);
                float r = std::cbrt(RandomFloat(0, 1)) * m_SphereRadius; // cbrt for uniform volume
                
                p.position.x += r * sin(phi) * cos(theta);
                p.position.y += r * sin(phi) * sin(theta);
                p.position.z += r * cos(phi);
            } else if (m_Shape == EmitterShape::Box) {
                p.position.x += RandomFloat(-m_BoxSize.x * 0.5f, m_BoxSize.x * 0.5f);
                p.position.y += RandomFloat(-m_BoxSize.y * 0.5f, m_BoxSize.y * 0.5f);
                p.position.z += RandomFloat(-m_BoxSize.z * 0.5f, m_BoxSize.z * 0.5f);
            }
            
            p.velocity = GetRandomVelocity();
            
            // Velocity Inheritance
            // Velocity Inheritance
            if (m_VelocityInheritance > 0.0f) {
                // Approximate emitter velocity from last frame position
                // If dt is small, velocity can be large, so we clamp or just take displacement.
                // Velocity = Displacement / dt.
                // However, we don't have dt here easily without changing signature.
                // Let's use m_LastPosition which is updated at end of Update().
                // Current m_Position - m_LastPosition is the displacement this frame.
                // If we assume this displacement happened over the last frame's dt, 
                // and we want to impart that VELOCITY, we need to know that dt.
                // BUT, simply adding the displacement to the particle's velocity (units/sec) is WRONG.
                // We need (Displacement / dt) * modifier.
                // Problem: SpawnParticle doesn't know dt.
                // Solution: Store 'm_CurrentVelocity' in Update() before calling SpawnParticle.
                
                // For now, let's use a stored member calculated in Update()
                p.velocity += m_CalculatedEmitterVelocity * m_VelocityInheritance;
            }
            
            p.size = m_SizeStart;
            p.color = m_ColorStart;
            p.age = 0.0f;
            p.lifetime = m_ParticleLifetime;
            p.lifeRatio = 0.0f;
            
            // Trails
            bool trailsEnabled = m_EnableTrails && m_ActiveLOD.enableTrails;
            if (trailsEnabled) {
                p.hasTrail = true;
                if (!p.trail) {
                    p.trail = std::make_shared<ParticleTrail>(m_TrailLength);
                }
                p.trail->Reset();
            } else {
                p.hasTrail = false;
            }
            
            break;
        }
    }
    }
}

void ParticleEmitter::UpdateCPU(float deltaTime) {
    for (auto& p : m_Particles) {
        if (!p.active) continue;
        
        p.age += deltaTime;
        if (p.age >= p.lifetime) {
            p.active = false;
            
            // Sub-Emitter on Death
            if (m_SubEmitterDeath) {
                m_SubEmitterDeath->SpawnAtPosition(p.position, 1); // Spawn 1 particle? or burst?
                // Let's spawn 1 for now, or maybe a burst.
                // If it's an explosion, usually a burst of X.
                // We'll trust the SubEmitter's Burst method logic if we called it, 
                // but SpawnAtPosition is single generic.
                // Let's assume SubEmitter is configured as an explosion (Burst) or trail (Stream).
                // If we want a burst, we should call BurstAtPosition?
                // For now, SpawnAtPosition(p, 5)?
            }
            
            if (p.hasTrail && p.trail) p.trail->Reset();
            continue;
        }
        
        p.lifeRatio = p.age / p.lifetime;
        
        // Visuals: Color & Size
        if (!m_ColorGradient.empty()) {
            p.color = EvaluateGradient(p.lifeRatio);
        } else {
            // Linear lerp existing
            float t = p.lifeRatio;
            p.color = m_ColorStart * (1.0f - t) + m_ColorEnd * t;
        }
        
        if (!m_SizeCurve.empty()) {
            p.size = EvaluateSize(p.lifeRatio);
        } else {
             float t = p.lifeRatio;
             p.size = m_SizeStart * (1.0f - t) + m_SizeEnd * t;
        }
        
        // Apply Forces
        p.velocity += m_Gravity * deltaTime;
        
        // Global Wind
        p.velocity += m_PhysicsContext.wind * deltaTime;
        
        // Attraction Points
        if (m_PhysicsContext.attractors) {
            const auto& attractors = *m_PhysicsContext.attractors;
            for (const auto& att : attractors) {
                Vec3 dir = att.pos - p.position;
                float distSq = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;
                if (distSq > 0.0001f) {
                    float dist = std::sqrt(distSq);
                    // Force = Direction * Strength * dt (simplified)
                    // F = ma -> a = F/m (assuming mass 1)
                    Vec3 force = (dir / dist) * (att.strength * deltaTime);
                    p.velocity += force;
                }
            }
        }

        // Apply Turbulence
        if (m_EnableTurbulence && m_ActiveLOD.enableTurbulence) {
            // Simple random turbulence
            // Better: Perlin noise
            // Placeholder:
             p.velocity.x += RandomFloat(-m_TurbulenceStrength, m_TurbulenceStrength) * deltaTime;
             p.velocity.y += RandomFloat(-m_TurbulenceStrength, m_TurbulenceStrength) * deltaTime;
             p.velocity.z += RandomFloat(-m_TurbulenceStrength, m_TurbulenceStrength) * deltaTime;
        }
        
        // Move
        p.position += p.velocity * deltaTime;
        
        // Update Trail
        if (p.hasTrail && p.trail) {
             p.trail->AddPoint(p.position, p.color, m_TrailWidth * p.size);
             p.trail->Update(deltaTime);
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
    return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
}

void ParticleEmitter::HandleShapeCollisions(Particle& particle, float deltaTime) {
    for (const auto& shape : m_CollisionShapes) {
        // Simple sphere collision for now
        // Assuming shape has GetPosition methods etc (Interface needed)
        // For now, skip implementation details, just placeholder logic
    }
    
    // Bounds check (Floor)
    if (particle.position.y < 0.0f) {
        particle.position.y = 0.0f;
        particle.velocity.y = -particle.velocity.y * m_DefaultRestitution;
        particle.velocity.x *= m_DefaultFriction;
        particle.velocity.z *= m_DefaultFriction;
    }
}

void ParticleEmitter::HandleParticleCollisions(float deltaTime) {
    BuildSpatialGrid();
    
    // Iterate over cells
    for (auto& cell : m_SpatialGrid.cells) {
        auto& particles = cell.second;
        for (size_t i = 0; i < particles.size(); ++i) {
            for (size_t j = i + 1; j < particles.size(); ++j) {
                ResolveParticleCollision(*particles[i], *particles[j]);
            }
        }
    }
}

void ParticleEmitter::BuildSpatialGrid() {
    m_SpatialGrid.cells.clear();
    m_SpatialGrid.cellSize = m_ParticleCollisionRadius * 2.0f;
    if (m_SpatialGrid.cellSize <= 0.0f) m_SpatialGrid.cellSize = 0.1f;
    
    for (auto& p : m_Particles) {
        if (!p.active) continue;
        int index = m_SpatialGrid.GetCellIndex(p.position);
        m_SpatialGrid.cells[index].push_back(&p);
    }
}

void ParticleEmitter::ResolveParticleCollision(Particle& p1, Particle& p2) {
    Vec3 diff = p1.position - p2.position;
    float distSq = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
    float radiusSum = m_ParticleCollisionRadius * 2.0f; // Simplified
    
    if (distSq < radiusSum * radiusSum && distSq > 0.0001f) {
        float dist = sqrt(distSq);
        float penetration = radiusSum - dist;
        Vec3 normal = diff * (1.0f / dist);
        
        // Separation
        Vec3 separation = normal * (penetration * 0.5f);
        p1.position = p1.position + separation;
        p2.position = p2.position - separation;
        
        // Bounce
        float v1Dot = p1.velocity.x*normal.x + p1.velocity.y*normal.y + p1.velocity.z*normal.z;
        float v2Dot = p2.velocity.x*normal.x + p2.velocity.y*normal.y + p2.velocity.z*normal.z;
        
        float vDiff = v1Dot - v2Dot;
        p1.velocity = p1.velocity - normal * (vDiff * 0.5f * (1.0f + m_DefaultRestitution));
        p2.velocity = p2.velocity + normal * (vDiff * 0.5f * (1.0f + m_DefaultRestitution));
    }
}

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
    m_GPUParticles.resize(m_MaxParticles);
    glGenBuffers(1, &m_ParticleSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ParticleSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, m_GPUParticles.size() * sizeof(GPUParticle), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    
    
    // Create atomic counter buffer
    glGenBuffers(1, &m_AtomicCounterBuffer);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_AtomicCounterBuffer);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

    // Create Sort SSBO
    m_SortBufferSize = NextPowerOfTwo(m_MaxParticles);
    glGenBuffers(1, &m_SortSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_SortSSBO);
    // Initialize with size for "struct { float distanceSq; uint index; }" -> 8 bytes
    // We initialize with empty data
    glBufferData(GL_SHADER_STORAGE_BUFFER, m_SortBufferSize * 8, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // Create Trail SSBO
    // Size: MaxParticles * MaxTrailLength * sizeof(Vec4) (Position + ?)
    // Let's store position (vec3) + width/reserved (float) = vec4
    if (m_EnableTrails) {
        if (m_TrailLength < 2) m_TrailLength = 10; // Default minimum
        size_t trailBufferSize = m_MaxParticles * m_TrailLength * 4 * sizeof(float); // vec4
        glGenBuffers(1, &m_TrailSSBO);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_TrailSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, trailBufferSize, nullptr, GL_DYNAMIC_DRAW);
        // Initialize with 0?
        // simple clear
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_RGBA32F, GL_RGBA, GL_FLOAT, nullptr); 
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        std::cout << "GPU Compute Trail Buffer initialized (Size: " << trailBufferSize << " bytes)" << std::endl;
    }
    
    std::cout << "GPU Compute initialized for particle system (Sort Buffer Size: " << m_SortBufferSize << ")" << std::endl;

    // Init Collision Logic
    InitGPUCollision();
}

void ParticleEmitter::ShutdownGPUCompute() {
    if (m_ParticleSSBO != 0) {
        glDeleteBuffers(1, &m_ParticleSSBO);
        m_ParticleSSBO = 0;
    }
    if (m_SortSSBO != 0) {
        glDeleteBuffers(1, &m_SortSSBO);
        m_SortSSBO = 0;
    }
    if (m_TrailSSBO != 0) {
        glDeleteBuffers(1, &m_TrailSSBO);
        m_TrailSSBO = 0;
    }
    if (m_AtomicCounterBuffer != 0) {
        glDeleteBuffers(1, &m_AtomicCounterBuffer);
        m_AtomicCounterBuffer = 0;
    }
    m_ComputeShader = nullptr;
    
    if (m_GridSSBO != 0) {
        glDeleteBuffers(1, &m_GridSSBO);
        m_GridSSBO = 0;
    }
    m_SpatialHashShader = nullptr;
    m_BitonicSortShader = nullptr;
    m_GridBuildShader = nullptr;
    m_CollisionShader = nullptr;
    m_SphDensityShader = nullptr;
    m_SphForceShader = nullptr;
    m_SpringForceShader = nullptr;
    m_ApplySpringForceShader = nullptr;
    m_IndirectArgsShader = nullptr;
    
    if (m_SpringSSBO != 0) {
        glDeleteBuffers(1, &m_SpringSSBO);
        m_SpringSSBO = 0;
    }
    if (m_ForceSSBO != 0) {
        glDeleteBuffers(1, &m_ForceSSBO);
        m_ForceSSBO = 0;
    }
    if (m_IndirectBuffer != 0) {
        glDeleteBuffers(1, &m_IndirectBuffer);
        m_IndirectBuffer = 0;
    }
}

void ParticleEmitter::UpdateGPU(float deltaTime) {
    if (!m_ComputeShader || m_ParticleSSBO == 0) return;
    
    // Upload particle data to GPU if NOT persistent (or first run)
    // For persistent mode, we assume data is already there or initialized
    if (!m_GPUPersistent) {
        // Convert CPU particles to GPU layout
        for (size_t i = 0; i < m_Particles.size(); ++i) {
            const auto& p = m_Particles[i];
            m_GPUParticles[i].position = Vec4(p.position.x, p.position.y, p.position.z, p.size);
            m_GPUParticles[i].velocity = Vec4(p.velocity.x, p.velocity.y, p.velocity.z, p.lifetime);
            m_GPUParticles[i].color = p.color;
            m_GPUParticles[i].properties = Vec4(p.age, p.lifetime > 0 ? p.lifeRatio : 0.0f, p.active ? 1.0f : 0.0f, 0.0f);
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ParticleSSBO);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, m_GPUParticles.size() * sizeof(GPUParticle), m_GPUParticles.data());
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
    m_ComputeShader->SetFloat("spawnRate", m_SpawnRate * m_ActiveLOD.emissionMultiplier);
    m_ComputeShader->SetFloat("time", m_Time);
    
    // Trail uniforms if trails are enabled
    bool enableTrails = m_EnableTrails && m_ActiveLOD.enableTrails;
    if (enableTrails && m_TrailSSBO != 0) {
        m_ComputeShader->SetInt("u_EnableTrails", 1);
        m_ComputeShader->SetInt("u_TrailLength", m_TrailLength);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_TrailSSBO); // Binding 2 for Trails
    } else {
        m_ComputeShader->SetInt("u_EnableTrails", 0);
        m_ComputeShader->SetInt("u_TrailLength", 0);
    }
    
    // Dispatch compute shader
    unsigned int numWorkGroups = (m_MaxParticles + 255) / 256;
    m_ComputeShader->Dispatch(numWorkGroups, 1, 1);
    
    // Memory barrier
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);
    
    // Perform Collision Detection if enabled
    if (m_EnableParticleCollisions || m_PhysicsMode == PhysicsMode::SPHFluid) {
        DispatchGPUCollision(deltaTime);
    }
    
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




void ParticleEmitter::InitGPUCollision() {
    // Load Shaders
    m_SpatialHashShader = std::make_unique<Shader>();
    if (!m_SpatialHashShader->LoadComputeShader("shaders/particle_spatial_hash.comp")) {
        std::cerr << "Failed to load particle spatial hash shader" << std::endl;
    }
    
    m_BitonicSortShader = std::make_unique<Shader>();
    if (!m_BitonicSortShader->LoadComputeShader("shaders/particle_bitonic_sort.comp")) {
        std::cerr << "Failed to load particle bitonic sort shader" << std::endl;
    }
    
    m_GridBuildShader = std::make_unique<Shader>();
    if (!m_GridBuildShader->LoadComputeShader("shaders/particle_grid_build.comp")) {
        std::cerr << "Failed to load particle grid build shader" << std::endl;
    }
    
    m_CollisionShader = std::make_unique<Shader>();
    if (!m_CollisionShader->LoadComputeShader("shaders/particle_resolve_collision.comp")) {
        std::cerr << "Failed to load particle resolve collision shader" << std::endl;
    }
    
    m_SphDensityShader = std::make_unique<Shader>();
    if (!m_SphDensityShader->LoadComputeShader("shaders/particle_sph_density.comp")) {
        std::cerr << "Failed to load particle sph density shader" << std::endl;
    }
    
    m_SphForceShader = std::make_unique<Shader>();
    if (!m_SphForceShader->LoadComputeShader("shaders/particle_sph_force.comp")) {
        std::cerr << "Failed to load particle sph force shader" << std::endl;
    }

    m_SpringForceShader = std::make_unique<Shader>();
    if (!m_SpringForceShader->LoadComputeShader("shaders/particle_spring_force.comp")) {
        std::cerr << "Failed to load particle spring force shader" << std::endl;
    }
    
    m_ApplySpringForceShader = std::make_unique<Shader>();
    if (!m_ApplySpringForceShader->LoadComputeShader("shaders/particle_apply_spring_force.comp")) {
        std::cerr << "Failed to load particle apply spring force shader" << std::endl;
    }
    
    m_IndirectArgsShader = std::make_unique<Shader>();
    if (!m_IndirectArgsShader->LoadComputeShader("shaders/particle_indirect_args.comp")) {
        std::cerr << "Failed to load particle indirect args shader" << std::endl;
    }
    
    // Indirect Draw Buffer (Atomic)
    glGenBuffers(1, &m_IndirectBuffer);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, m_IndirectBuffer);
    // DrawArraysIndirectCommand: 4 * uint = 16 bytes
    glBufferData(GL_DRAW_INDIRECT_BUFFER, 4 * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

    // Create Grid Buffer
    // Size: GridSize * 2 (Start, End) * sizeof(uint)
    glGenBuffers(1, &m_GridSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_GridSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, m_CollisionGridSize * 2 * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void ParticleEmitter::DispatchGPUCollision(float deltaTime) {
    if (!m_SpatialHashShader || !m_BitonicSortShader || !m_GridBuildShader || !m_CollisionShader) return;
    
    // 1. Spatial Hash
    // Input: Particles, Output: SortBuffer (Keys)
    m_SpatialHashShader->Use();
    m_SpatialHashShader->SetInt("u_MaxParticles", m_MaxParticles);
    m_SpatialHashShader->SetFloat("u_CellSize", m_CollisionCellSize);
    m_SpatialHashShader->SetInt("u_GridSize", m_CollisionGridSize);
    
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ParticleSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_SortSSBO); // Initializing SortBuffer with Hashes
    
    unsigned int numWorkGroups = (m_MaxParticles + 255) / 256;
    m_SpatialHashShader->Dispatch(numWorkGroups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    // 2. Sort (Bitonic)
    // We sort the SortBuffer based on Hashes
    m_BitonicSortShader->Use();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_SortSSBO);
    
    // Reuse NextPowerOfTwo from file scope
    unsigned int n = NextPowerOfTwo(m_MaxParticles); 
    
    for (unsigned int k = 2; k <= n; k <<= 1) {
        for (unsigned int j = k >> 1; j > 0; j >>= 1) {
            m_BitonicSortShader->SetInt("j", j);
            m_BitonicSortShader->SetInt("k", k);
            m_BitonicSortShader->Dispatch(n / 256, 1, 1); // We launch Threads = N/2. Wait, shader is N sized.
            // My shader logic: if (ixj > i) return; 
            // This means I launch N threads, and half of them return.
            // So Dispatch(n / 256) is correct if n is multiple of 256. 
            // n is NextPowerOfTwo(MaxParticles).
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
    }
    
    // 3. Build Grid (Find Start/End)
    // Reset Grid Buffer first? Or shader handles it?
    // Compute shader writes only if it finds boundaries.
    // If a cell is empty, it remains garbage?
    // We MUST Init Grid Buffer to -1 (Empty).
    // Use glClearBufferData or a simple Compute Clear.
    // Let's use glClearBufferData for simplicity.
    unsigned int val = 0xFFFFFFFF;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_GridSSBO);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &val); // Clear to All 1s
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT); // Ensure clear is done
    
    m_GridBuildShader->Use();
    m_GridBuildShader->SetInt("u_NumParticles", m_MaxParticles);
    
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_SortSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_GridSSBO);
    
    m_GridBuildShader->Dispatch(numWorkGroups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    m_GridBuildShader->Dispatch(numWorkGroups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    // 4. Resolve Physics based on Mode
    if (m_PhysicsMode == PhysicsMode::SPHFluid) {
        // SPH Density Pass
        m_SphDensityShader->Use();
        m_SphDensityShader->SetInt("u_MaxParticles", m_MaxParticles);
        m_SphDensityShader->SetInt("u_GridSize", m_CollisionGridSize);
        m_SphDensityShader->SetFloat("u_CellSize", m_CollisionCellSize);
        m_SphDensityShader->SetFloat("u_SmoothingRadius", m_SmoothingRadius);
        // Poly6 Coefficient: 315 / (64 * PI * h^9)
        float h = m_SmoothingRadius;
        float h9 = static_cast<float>(std::pow(h, 9));
        float poly6 = 315.0f / (64.0f * 3.14159f * h9);
        m_SphDensityShader->SetFloat("u_Poly6Coef", poly6);
        
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ParticleSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_SortSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_GridSSBO);
        
        m_SphDensityShader->Dispatch(numWorkGroups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        
        // SPH Force Pass
        m_SphForceShader->Use();
        m_SphForceShader->SetInt("u_MaxParticles", m_MaxParticles);
        m_SphForceShader->SetInt("u_GridSize", m_CollisionGridSize);
        m_SphForceShader->SetFloat("u_CellSize", m_CollisionCellSize);
        m_SphForceShader->SetFloat("u_SmoothingRadius", m_SmoothingRadius);
        
        // Spiky Gradient Coefficient: -45 / (PI * h^6)
        float h6 = static_cast<float>(std::pow(h, 6));
        float spiky = -45.0f / (3.14159f * h6);
        m_SphForceShader->SetFloat("u_SpikyCoef", spiky);
        m_SphForceShader->SetFloat("u_ViscosityCoef", -spiky); // Viscosity kernel often uses 45, spiky uses -45.
                                                               // Using +45 for Laplacian approximation here.
        
        m_SphForceShader->SetFloat("u_RestDensity", m_RestDensity);
        m_SphForceShader->SetFloat("u_GasConstant", m_GasConstant);
        m_SphForceShader->SetFloat("u_Viscosity", m_Viscosity);
        m_SphForceShader->SetFloat("u_DeltaTime", deltaTime);
        
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ParticleSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_SortSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_GridSSBO);
        
        m_SphForceShader->Dispatch(numWorkGroups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        
        // After Forces, we might also want standard collision resolution to keep them in bounds or apart?
        // SPH handles separation via pressure, but boundaries are needed.
        // For now, let's allow standard collision to run IF specifically enabled together with SPH,
        // but typically SPH replaces it.
        // If m_EnableParticleCollisions is true, we run that too.
    }
    
    // Standard Collisions (Elastic sphere-sphere)
    if (m_EnableParticleCollisions) {
        m_CollisionShader->Use();
        m_CollisionShader->SetInt("u_MaxParticles", m_MaxParticles);
        m_CollisionShader->SetInt("u_GridSize", m_CollisionGridSize);
        m_CollisionShader->SetFloat("u_CellSize", m_CollisionCellSize);
        m_CollisionShader->SetFloat("u_CollisionRadius", m_ParticleCollisionRadius); // Assuming this is set
        m_CollisionShader->SetFloat("u_Restitution", m_DefaultRestitution);
        m_CollisionShader->SetFloat("u_Friction", m_DefaultFriction);
        
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ParticleSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_SortSSBO);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_GridSSBO);
        
        m_CollisionShader->Dispatch(numWorkGroups, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    
    // 5. Spring Physics (Cloth/SoftBody)
    if (m_PhysicsMode == PhysicsMode::Cloth || m_PhysicsMode == PhysicsMode::SoftBody) {
        if (m_SpringSSBO == 0 && !m_Springs.empty()) {
            // Upload Springs (Once)
            glGenBuffers(1, &m_SpringSSBO);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_SpringSSBO);
            glBufferData(GL_SHADER_STORAGE_BUFFER, m_Springs.size() * sizeof(Spring), m_Springs.data(), GL_STATIC_DRAW);
            
            // Create Force Accumulator
            // MaxParticles * 3 * sizeof(int)
            glGenBuffers(1, &m_ForceSSBO);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ForceSSBO);
            glBufferData(GL_SHADER_STORAGE_BUFFER, m_MaxParticles * 3 * sizeof(int), nullptr, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            
            // Clear Force Buffer Initially
            int zero = 0;
            glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED_INTEGER, GL_INT, &zero);
        }
        
        if (m_SpringSSBO != 0) {
            // A. Calculate Spring Forces
            m_SpringForceShader->Use();
            m_SpringForceShader->SetInt("u_NumSprings", (int)m_Springs.size());
            m_SpringForceShader->SetFloat("u_DeltaTime", deltaTime);
            
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ParticleSSBO);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_SpringSSBO);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_ForceSSBO);
            
            unsigned int numGroupsSprings = ((unsigned int)m_Springs.size() + 255) / 256;
            m_SpringForceShader->Dispatch(numGroupsSprings, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
            // B. Apply Forces
            m_ApplySpringForceShader->Use();
            m_ApplySpringForceShader->SetInt("u_MaxParticles", m_MaxParticles);
            m_ApplySpringForceShader->SetFloat("u_DeltaTime", deltaTime);
            m_ApplySpringForceShader->SetFloat("u_Mass", m_DefaultMass); // User set mass
            
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ParticleSSBO);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_ForceSSBO);
            
            m_ApplySpringForceShader->Dispatch(numWorkGroups, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
    }
    
    // 6. Update Indirect Draw Args
    if (m_IndirectBuffer != 0 && m_IndirectArgsShader) {
        m_IndirectArgsShader->Use();
        glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 1, m_AtomicCounterBuffer); // Matches shader Binding 1
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_IndirectBuffer);      // Matches shader Binding 6
        
        m_IndirectArgsShader->Dispatch(1, 1, 1);
        glMemoryBarrier(GL_COMMAND_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    }
}

void ParticleEmitter::InitCloth(int width, int height, float spacing, float stiffness, float damping) {
    m_PhysicsMode = PhysicsMode::Cloth;
    m_MaxParticles = width * height;
    m_Particles.resize(m_MaxParticles);
    m_GPUParticles.resize(m_MaxParticles);
    m_Springs.clear();
    
    // Spawn Grid
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int i = y * width + x;
            Vec3 pos = m_Position + Vec3((float)x * spacing, (float)y * -spacing, 0); // Hang down
            
            // Init Particle
            m_Particles[i].position = pos;
            m_Particles[i].velocity = Vec3(0,0,0);
            m_Particles[i].color = m_ColorStart;
            m_Particles[i].size = m_SizeStart;
            m_Particles[i].age = 0;
            m_Particles[i].lifetime = 99999.0f; // Infinite
            m_Particles[i].active = true;
            m_Particles[i].lifeRatio = 1.0f; // Hack to keep alive
            
            // Structural Springs (Right and Down)
            if (x < width - 1) {
                // Right
                Spring s;
                s.p1 = i;
                s.p2 = y * width + (x + 1);
                s.restLength = spacing;
                s.stiffness = stiffness;
                s.damping = damping;
                m_Springs.push_back(s);
            }
            if (y < height - 1) {
                // Down
                Spring s;
                s.p1 = i;
                s.p2 = (y + 1) * width + x;
                s.restLength = spacing;
                s.stiffness = stiffness;
                s.damping = damping;
                m_Springs.push_back(s);
            }
            // Shear Springs (Diagonals)
            if (x < width - 1 && y < height - 1) {
                Spring s;
                s.p1 = i;
                s.p2 = (y + 1) * width + (x + 1);
                s.restLength = spacing * 1.414f;
                s.stiffness = stiffness;
                s.damping = damping;
                m_Springs.push_back(s);
                
                s.p1 = y * width + (x + 1);
                s.p2 = (y + 1) * width + x;
                m_Springs.push_back(s);
            }
            // Bend Springs (Skip 1)
            if (x < width - 2) {
                Spring s;
                s.p1 = i;
                s.p2 = y * width + (x + 2);
                s.restLength = spacing * 2.0f;
                s.stiffness = stiffness;
                s.damping = damping;
                m_Springs.push_back(s);
            }
            if (y < height - 2) {
                Spring s;
                s.p1 = i;
                s.p2 = (y + 2) * width + x;
                s.restLength = spacing * 2.0f;
                s.stiffness = stiffness;
                s.damping = damping;
                m_Springs.push_back(s);
            }
        }
    }
    
    SetUseGPUCompute(true); // Auto enable GPU
}

void ParticleEmitter::InitSoftBody(int width, int height, int depth, float spacing, float stiffness, float damping) {
    m_PhysicsMode = PhysicsMode::SoftBody;
    m_MaxParticles = width * height * depth;
    m_Particles.resize(m_MaxParticles);
    m_GPUParticles.resize(m_MaxParticles);
    m_Springs.clear();
    
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int i = z * (width * height) + y * width + x;
                Vec3 pos = m_Position + Vec3((float)x * spacing, (float)y * spacing, (float)z * spacing);
                
                m_Particles[i].position = pos;
                m_Particles[i].velocity = Vec3(0,0,0);
                m_Particles[i].color = m_ColorStart;
                m_Particles[i].size = m_SizeStart;
                m_Particles[i].active = true;
                m_Particles[i].lifetime = 99999.0f;
                
                // Structural Neighbors: +X, +Y, +Z
                // X
                if (x < width - 1) {
                    Spring s;
                    s.p1 = i;
                    s.p2 = z * (width * height) + y * width + (x + 1);
                    s.restLength = spacing;
                    s.stiffness = stiffness;
                    s.damping = damping;
                    m_Springs.push_back(s);
                }
                // Y
                if (y < height - 1) {
                    Spring s;
                    s.p1 = i;
                    s.p2 = z * (width * height) + (y + 1) * width + x;
                    s.restLength = spacing;
                    s.stiffness = stiffness;
                    s.damping = damping;
                    m_Springs.push_back(s);
                }
                // Z
                if (z < depth - 1) {
                    Spring s;
                    s.p1 = i;
                    s.p2 = (z + 1) * (width * height) + y * width + x;
                    s.restLength = spacing;
                    s.stiffness = stiffness;
                    s.damping = damping;
                    m_Springs.push_back(s);
                }
                
                // Shear and Bend could be added for stability
            }
        }
    }
    
    SetUseGPUCompute(true);
}

// Visual Helpers
void ParticleEmitter::AddGradientStop(float t, const Vec4& color) {
    GradientStop stop = {t, color};
    m_ColorGradient.push_back(stop);
    // Sort by t
    std::sort(m_ColorGradient.begin(), m_ColorGradient.end(), [](const GradientStop& a, const GradientStop& b) {
        return a.t < b.t;
    });
}

void ParticleEmitter::AddSizeCurvePoint(float size) {
    m_SizeCurve.push_back(size);
}

Vec4 ParticleEmitter::EvaluateGradient(float t) const {
    if (m_ColorGradient.empty()) return m_ColorStart;
    if (m_ColorGradient.size() == 1) return m_ColorGradient[0].color;
    
    // Find segment
    for (size_t i = 0; i < m_ColorGradient.size() - 1; ++i) {
        if (t >= m_ColorGradient[i].t && t <= m_ColorGradient[i+1].t) {
            float range = m_ColorGradient[i+1].t - m_ColorGradient[i].t;
            if (range < 0.0001f) return m_ColorGradient[i].color;
            float localT = (t - m_ColorGradient[i].t) / range;
            return m_ColorGradient[i].color * (1.0f - localT) + m_ColorGradient[i+1].color * localT;
        }
    }
    
    // Clamping
    if (t < m_ColorGradient.front().t) return m_ColorGradient.front().color;
    return m_ColorGradient.back().color;
}

float ParticleEmitter::EvaluateSize(float t) const {
    if (m_SizeCurve.empty()) return m_SizeStart;
    if (m_SizeCurve.size() == 1) return m_SizeCurve[0];
    
    // Equidistant points assumed 0..1
    float step = 1.0f / (float)(m_SizeCurve.size() - 1); // e.g. 5 points -> step 0.25 (0, 0.25, 0.5, 0.75, 1.0)
    int index = (int)(t / step);
    if (index < 0) index = 0;
    if (index >= (int)m_SizeCurve.size() - 1) return m_SizeCurve.back();
    
    float localT = (t - (index * step)) / step;
    return m_SizeCurve[index] * (1.0f - localT) + m_SizeCurve[index+1] * localT;
}

void ParticleEmitter::SpawnAtPosition(const Vec3& pos, int count) {
    if (!m_Active) return;
    
    // Backup original pos
    Vec3 originalPos = m_Position;
    m_Position = pos; // Temporarily move emitter
    
    for(int i=0; i<count; ++i) {
        SpawnParticle();
    }
    
    m_Position = originalPos; // Restore
}

void ParticleEmitter::SetMaxParticles(int count) {
    if (count <= 0 || count == m_MaxParticles) return;
    
    // Resize CPU vectors
    if (count < m_MaxParticles) {
        // Shrinking: Just resize, losing end particles
        m_Particles.resize(count);
        // Also GPUParticles if they exist CPU side
        if (m_GPUParticles.size() > count) m_GPUParticles.resize(count);
    } else {
        // Growing
        m_Particles.resize(count);
        m_GPUParticles.resize(count);
    }
    
    m_MaxParticles = count;
    
    // If using GPU Compute, we must re-allocate buffers
    if (m_UseGPUCompute) {
        // Full shutdown and re-init is safest but slow. 
        // For resize, we might want to preserve data, but that's complex (readback -> init -> upload).
        // For now, let's just re-init which resets simulation (easiest safe path)
        ShutdownGPUCompute();
        InitGPUCompute();
    }
}

int ParticleEmitter::KillOldest(int count) {
    int killed = 0;
    
    // In CPU mode, find oldest active particles
    // "Oldest" usually means highest Age.
    // Or we could just kill *any* active particle to free space.
    // Making space is usually for spawning new ones.
    
    // optimization: just kill the first 'count' active particles we find?
    // Or actually sort by age? Sorting is expensive.
    
    // Let's iterate and kill active ones with highest age fraction
    // Actually, simple "kill random/first" is often acceptable for high load shedding.
    
    for (auto& p : m_Particles) {
        if (killed >= count) break;
        if (p.active) {
            p.active = false;
            p.age = p.lifetime; // expire it
            killed++;
        }
    }
    
    // If GPU Computer, we can't easily kill specific ones without complex compute shader logic.
    // We'd need to atomic decrement count or flag them.
    // For now, KillOldest only works reliably on CPU sim. 
    // On GPU, maybe we just lower the "ActiveParticleCount" uniform if we could?
    
    return killed;
}
