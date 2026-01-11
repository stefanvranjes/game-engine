#include "FoamParticleSystem.h"
#include "FluidParticle.h"
#include "PhysXBackend.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <tuple>
#include <functional>

FoamParticleSystem::FoamParticleSystem()
    : m_MaxParticles(10000)
    , m_DragCoefficient(0.5f)
    , m_Buoyancy(0.1f)
    , m_FoamLifetime(2.0f)
    , m_SprayLifetime(1.0f)
    , m_BubbleLifetime(3.0f)
    , m_AnimationSpeed(1.0f)
    , m_SyncAnimationToLifetime(true)  // Default to synced for better dissolution
    , m_MergeRadius(0.0f) // Disabled by default
    , m_SpawnAccumulator(0.0f)
    , m_TotalEmitted(0)
{
}

FoamParticleSystem::~FoamParticleSystem() {
}

// Define helper collision method
void FoamParticleSystem::Update(float deltaTime, const Vec3& gravity, PhysXBackend* backend) {
    // Generate new particles if needed
    // Update existing particles
    for (auto& particle : m_Particles) {
        if (!particle.active) continue;
        
        UpdateParticle(particle, deltaTime, gravity);
        
        // Handle collision and adhesion
        if (backend && m_SurfaceAdhesion > 0.0f) {
            // Simplified raycast for optimization: Only check if moving fast enough or near ground
            // Or just check downwards/forward?
            
            // Cast ray in velocity direction
            Vec3 dir = particle.velocity;
            float speed = dir.Length();
            if (speed > 0.1f) {
                dir = dir / speed; // Normalize
                float checkDist = speed * deltaTime * 1.5f; // Look ahead
                checkDist = std::max(checkDist, particle.size * 2.0f); // Minimum radius check
                
                RaycastHit hit;
                // Raycast from current position
                if (backend->Raycast(particle.position, particle.position + dir * checkDist, hit)) {
                    // Collision detected!
                    
                    // Adhesion logic:
                    // Stop or slide along surface
                    
                    // 1. Move to hit point (prevent tunneling)
                    particle.position = hit.position + hit.normal * 0.01f; // Offset slightly
                    
                    // 2. Reflect or slide velocity?
                    // Adhesion -> Kill normal velocity (stick)
                    // Project velocity onto plane: v_tangent = v - (v . n) * n
                    float vDotN = particle.velocity.Dot(hit.normal);
                    Vec3 tangentVel = particle.velocity - hit.normal * vDotN;
                    
                    // Apply friction/adhesion
                    // If adhesion is high, we stop completely.
                    // If adhesion is low, we slide.
                    // m_SurfaceAdhesion 0.0 -> Slide freely (0 friction)
                    // m_SurfaceAdhesion 1.0 -> Stick completely (infinite friction)
                    
                    particle.velocity = tangentVel * (1.0f - m_SurfaceAdhesion);
                    
                    // Also kill remaining velocity if very low to prevent jitter on slopes
                    if (particle.velocity.LengthSquared() < 0.01f) {
                        particle.velocity = Vec3(0, 0, 0);
                    }
                }
            }
        }
    }
    
    // Merge particles
    if (m_MergeRadius > 0.0f) {
        MergeParticles(m_MergeRadius);
    }
    
    // Remove dead particles
    RemoveDeadParticles();
    
    // Enforce limits
    EnforceParticleLimit();
}

void FoamParticleSystem::GenerateFromFluid(const std::vector<FluidParticle>& fluidParticles,
                                          const std::vector<FluidType>& fluidTypes,
                                          float velocityThreshold,
                                          float spawnRate) {
    if (static_cast<int>(m_Particles.size()) >= m_MaxParticles) {
        return;  // At capacity
    }
    
    // Count particles with high potential (Velocity + Aeration)
    int potentialCount = 0;
    
    // Aeration parameters
    const float minDensityRatio = 0.8f; // Below this ratio, fluid is aerated (80% of rest density)
    
    for (const auto& fp : fluidParticles) {
        if (!fp.active) continue;
        
        bool spawn = false;
        
        // 1. Velocity potential (Weber number / Turbulence)
        float speed = fp.velocity.Length();
        if (speed > velocityThreshold) {
            spawn = true;
        }
        
        // 2. Aeration potential (Trapped air / Low density)
        // If density is significantly lower than rest density, it implies trapped air
        if (!spawn && !fluidTypes.empty()) {
            float restDensity = fluidTypes[fp.fluidType].restDensity;
            if (restDensity > 0.0f) {
                float ratio = fp.density / restDensity;
                if (ratio < minDensityRatio) {
                    spawn = true;
                }
            }
        }
        
        if (spawn) {
            potentialCount++;
        }
    }
    
    // Calculate particles to spawn
    float particlesToSpawn = potentialCount * spawnRate;
    m_SpawnAccumulator += particlesToSpawn;
    
    int spawnCount = static_cast<int>(m_SpawnAccumulator);
    m_SpawnAccumulator -= spawnCount;
    
    // Spawn particles from random high-potential fluid particles
    int spawned = 0;
    const float minDensityRatio = 0.8f;
    
    for (const auto& fp : fluidParticles) {
        if (spawned >= spawnCount) break;
        if (!fp.active) continue;
        
        bool spawn = false;
        float speed = fp.velocity.Length();
        
        // Check velocity
        if (speed > velocityThreshold) {
            spawn = true;
        }
        
        // Check density
        if (!spawn && !fluidTypes.empty()) {
            float restDensity = fluidTypes[fp.fluidType].restDensity;
            if (restDensity > 0.0f) {
                float ratio = fp.density / restDensity;
                if (ratio < minDensityRatio) {
                    spawn = true;
                }
            }
        }
        
        if (spawn) {
            // Determine particle type based on velocity and density
            // If spawning due to low density but low velocity -> Bubble
            FoamParticle::Type type = FoamParticle::Type::Bubble;
            if (speed > velocityThreshold) {
                type = DetermineParticleType(fp.velocity);
            } else {
                type = FoamParticle::Type::Bubble; // Aerated water bubbles up
            }
            
            // Set lifetime based on type
            float lifetime;
            switch (type) {
                case FoamParticle::Type::Foam:
                    lifetime = m_FoamLifetime;
                    break;
                case FoamParticle::Type::Spray:
                    lifetime = m_SprayLifetime;
                    break;
                case FoamParticle::Type::Bubble:
                    lifetime = m_BubbleLifetime;
                    break;
            }
            
            // Add random variation to lifetime
            lifetime *= (0.8f + 0.4f * (static_cast<float>(rand()) / RAND_MAX));
            
            // Create foam particle with slight random offset
            Vec3 offset(
                (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.05f,
                (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.05f,
                (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.05f
            );
            
            Vec3 position = fp.position + offset;
            Vec3 velocity = fp.velocity * 0.8f;  // Slightly slower than fluid
            
            FoamParticle particle(position, velocity, type, lifetime);
            
            // Random start time for animation to prevent synchronization
            particle.animationTime = static_cast<float>(rand()) / RAND_MAX;
            
            // Assign texture based on type
            switch (type) {
                case FoamParticle::Type::Foam:
                    particle.textureIndex = 0;  // Irregular foam texture
                    break;
                case FoamParticle::Type::Spray:
                    particle.textureIndex = 2;  // Spray droplets
                    break;
                case FoamParticle::Type::Bubble:
                    // Randomly choose between bubble cluster (1) or large bubble (3)
                    particle.textureIndex = (rand() % 2 == 0) ? 1 : 3;
                    break;
            }
            
            m_Particles.push_back(particle);
            
            spawned++;
            
            // Check if we've hit the limit
            if (static_cast<int>(m_Particles.size()) >= m_MaxParticles) {
                break;
            }
        }
    }
}

void FoamParticleSystem::Clear() {
    m_Particles.clear();
    m_SpawnAccumulator = 0.0f;
}

int FoamParticleSystem::GetActiveParticleCount() const {
    int count = 0;
    for (const auto& p : m_Particles) {
        if (p.active) count++;
    }
    return count;
}

void FoamParticleSystem::UpdateParticle(FoamParticle& particle, float deltaTime, const Vec3& gravity) {
    // Decrease lifetime
    particle.lifetime -= deltaTime;
    if (particle.lifetime <= 0.0f) {
        particle.active = false;
        return;
    }
    
    // Apply forces based on particle type
    Vec3 force = gravity;
    
    switch (particle.type) {
        case FoamParticle::Type::Foam:
            // Foam floats on surface, minimal gravity
            force = force * 0.1f;
            break;
            
        case FoamParticle::Type::Spray:
            // Spray follows ballistic trajectory
            // Full gravity
            break;
            
        case FoamParticle::Type::Bubble:
            // Bubbles rise (negative buoyancy)
            force = force + Vec3(0, m_Buoyancy, 0);
            break;
    }
    
    // Apply drag
    Vec3 drag = particle.velocity * (-m_DragCoefficient);
    force = force + drag;
    
    // Integrate velocity
    particle.velocity = particle.velocity + force * deltaTime;
    
    // Integrate position
    particle.position = particle.position + particle.velocity * deltaTime;
    
    // Update animation time
    if (m_SyncAnimationToLifetime) {
        // Map 0..1 animation to full lifetime (Birth -> Death)
        // 1.0 = Birth (Start), 0.0 = Death (End)
        // We want Animation 0.0 at Birth, 1.0 at Death
        particle.animationTime = 1.0f - particle.GetLifeRatio();
    } else {
        // Looping animation
        particle.animationTime += deltaTime * m_AnimationSpeed;
        if (particle.animationTime > 1.0f) {
            particle.animationTime = fmod(particle.animationTime, 1.0f);
        }
    }
    
    // Fade size based on lifetime
    float lifeRatio = particle.GetLifeRatio();
    particle.size = 0.02f * lifeRatio;
}

void FoamParticleSystem::RemoveDeadParticles() {
    m_Particles.erase(
        std::remove_if(m_Particles.begin(), m_Particles.end(),
            [](const FoamParticle& p) { return !p.active; }),
        m_Particles.end()
    );
}

void FoamParticleSystem::EnforceParticleLimit() {
    while (static_cast<int>(m_Particles.size()) > m_MaxParticles) {
        // Remove oldest particle (first in vector)
        m_Particles.erase(m_Particles.begin());
    }
}

FoamParticle::Type FoamParticleSystem::DetermineParticleType(const Vec3& velocity) {
    float speed = velocity.Length();
    float verticalComponent = velocity.y;
    
    // High upward velocity -> spray
    if (verticalComponent > 2.0f && speed > 3.0f) {
        return FoamParticle::Type::Spray;
    }
    
    // Downward or slow -> foam
    if (verticalComponent < 0.5f) {
        return FoamParticle::Type::Foam;
    }
    
    // Moderate upward -> bubble
    return FoamParticle::Type::Bubble;
}

#include <unordered_map>

// Simple hash for 3D integer coordinates
struct GridHash {
    std::size_t operator()(const std::tuple<int, int, int>& key) const {
        // Cantor pairing or similar, or just bit shifting
        // Simple hash suitable for typical grid ranges
        auto [x, y, z] = key;
        size_t h = 0;
        h ^= std::hash<int>{}(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(y) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(z) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

void FoamParticleSystem::MergeParticles(float radius) {
    if (radius <= 0.0001f) return;
    
    float cellSize = radius;
    std::unordered_map<std::tuple<int, int, int>, std::vector<int>, GridHash> grid;
    
    // Hash particles into grid
    for (int i = 0; i < m_Particles.size(); ++i) {
        if (!m_Particles[i].active) continue;
        
        const Vec3& pos = m_Particles[i].position;
        int cx = static_cast<int>(std::floor(pos.x / cellSize));
        int cy = static_cast<int>(std::floor(pos.y / cellSize));
        int cz = static_cast<int>(std::floor(pos.z / cellSize));
        
        grid[{cx, cy, cz}].push_back(i);
    }
    
    // Merge within cells
    // Note: We only check within same cell for simplicity/speed (aggressive optimization)
    // To be more accurate we'd check neighbors, but for foam clumping this is usually sufficient visual approx.
    for (auto& pair : grid) {
        std::vector<int>& indices = pair.second;
        if (indices.size() < 2) continue;
        
        // Brute force within cell
        for (size_t i = 0; i < indices.size(); ++i) {
            int idxA = indices[i];
            if (!m_Particles[idxA].active) continue;
            
            for (size_t j = i + 1; j < indices.size(); ++j) {
                int idxB = indices[j];
                if (!m_Particles[idxB].active) continue;
                
                // Only merge same type
                if (m_Particles[idxA].type != m_Particles[idxB].type) continue;
                
                // Check distance
                Vec3 diff = m_Particles[idxA].position - m_Particles[idxB].position;
                if (diff.LengthSquared() < radius * radius) {
                    // Merge B into A
                    FoamParticle& pA = m_Particles[idxA];
                    FoamParticle& pB = m_Particles[idxB];
                    
                    // Conservation - weighted average by mass-equivalent (size or life?)
                    // Let's use simple averaging for foam
                    
                    // New position (midpoint)
                    pA.position = (pA.position + pB.position) * 0.5f;
                    
                    // New velocity (momentum conservation)
                    pA.velocity = (pA.velocity + pB.velocity) * 0.5f;
                    
                    // Increase size (area conservation -> radius * sqrt(2))? 
                    // Or just slight increase to avoid explosion.
                    pA.size = std::max(pA.size, pB.size) * 1.1f;
                    
                    // Maximum lifetime wins (keeps foam alive)
                    pA.lifetime = std::max(pA.lifetime, pB.lifetime);
                    // Reset max lifetime to ratio is preserved roughly? 
                    // No, let it dissolve naturally.
                    
                    // Kill particle B
                    pB.active = false;
                }
            }
        }
    }
}