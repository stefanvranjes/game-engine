#include "PBDFluidSolver.h"
#include "PhysXBackend.h"
#include <cmath>
#include <algorithm>
#include <execution>
#include <atomic>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

PBDFluidSolver::PBDFluidSolver()
    : m_KernelRadius(0.1f)
    , m_SolverIterations(3)
    , m_Gravity(0.0f, -9.81f, 0.0f)
    , m_BoundaryMin(-10.0f, -10.0f, -10.0f)
    , m_BoundaryMax(10.0f, 10.0f, 10.0f)
    , m_EnableBoundary(true)
    , m_RelaxationEpsilon(600.0f)
    , m_ViscosityScale(1.0f)
    , m_SurfaceTensionScale(1.0f)
    , m_SpatialGrid(std::make_unique<SpatialHashGrid>())
    , m_PhysicsBackend(nullptr)
{
    memset(&m_Stats, 0, sizeof(Statistics));
}

PBDFluidSolver::~PBDFluidSolver() {
}

void PBDFluidSolver::Initialize(PhysXBackend* backend) {
    m_PhysicsBackend = backend;
}

void PBDFluidSolver::Update(std::vector<FluidParticle>& particles, 
                            const std::vector<FluidType>& fluidTypes,
                            float deltaTime) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Count active particles
    m_Stats.activeParticles = 0;
    for (const auto& p : particles) {
        if (p.active) m_Stats.activeParticles++;
    }
    
    if (m_Stats.activeParticles == 0) {
        return;
    }
    
    // Step 1: Apply external forces (gravity, user forces)
    auto t0 = std::chrono::high_resolution_clock::now();
    ApplyExternalForces(particles, fluidTypes, deltaTime);
    
    // Step 2: Predict positions
    PredictPositions(particles, fluidTypes, deltaTime);
    auto t1 = std::chrono::high_resolution_clock::now();
    m_Stats.predictTimeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
    
    // Step 3: Build spatial grid for neighbor search
    BuildSpatialGrid(particles);
    
    // Step 4: Find neighbors
    FindNeighbors(particles);
    auto t2 = std::chrono::high_resolution_clock::now();
    m_Stats.neighborSearchTimeMs = std::chrono::duration<float, std::milli>(t2 - t1).count();
    
    // Step 5: Solve constraints iteratively
    SolveConstraints(particles, fluidTypes, m_SolverIterations);
    auto t3 = std::chrono::high_resolution_clock::now();
    m_Stats.constraintTimeMs = std::chrono::duration<float, std::milli>(t3 - t2).count();
    
    // Step 6: Update velocities
    UpdateVelocities(particles, deltaTime);
    
    // Step 7: Apply viscosity
    ApplyViscosity(particles, fluidTypes);
    
    // Step 8: Apply surface tension
    ApplySurfaceTension(particles, fluidTypes);
    auto t4 = std::chrono::high_resolution_clock::now();
    m_Stats.velocityUpdateTimeMs = std::chrono::duration<float, std::milli>(t4 - t3).count();
    
    // Step 9: Handle collisions
    if (m_EnableBoundary) {
        HandleBoundaryCollisions(particles);
    }
    if (m_PhysicsBackend) {
        HandleRigidBodyCollisions(particles);
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    m_Stats.totalTimeMs = std::chrono::duration<float, std::milli>(endTime - startTime).count();
}

void PBDFluidSolver::ApplyExternalForces(std::vector<FluidParticle>& particles, 
                                        const std::vector<FluidType>& fluidTypes,
                                        float dt) {
    for (auto& p : particles) {
        if (!p.active) continue;
        
        // Apply gravity
        p.force = m_Gravity * fluidTypes[p.fluidType].particleMass;
        
        // Additional forces can be added here
    }
}

void PBDFluidSolver::PredictPositions(std::vector<FluidParticle>& particles, 
                                      const std::vector<FluidType>& fluidTypes,
                                      float dt) {
    std::for_each(std::execution::par, particles.begin(), particles.end(), 
        [&](FluidParticle& p) {
            if (!p.active) return;
            
            // Update velocity with forces
            p.velocity = p.velocity + (p.force / fluidTypes[p.fluidType].particleMass) * dt;
            
            // Predict position
            p.predictedPosition = p.position + p.velocity * dt;
            
            // Reset delta position
            p.deltaPosition = Vec3(0, 0, 0);
        }
    );
}

void PBDFluidSolver::BuildSpatialGrid(const std::vector<FluidParticle>& particles) {
    // Extract positions for spatial grid
    std::vector<Vec3> positions;
    positions.reserve(particles.size());
    
    for (const auto& p : particles) {
        if (p.active) {
            positions.push_back(p.predictedPosition);
        }
    }
    
    // Build grid with cell size = kernel radius
    m_SpatialGrid->Build(positions, m_KernelRadius);
}

void PBDFluidSolver::FindNeighbors(std::vector<FluidParticle>& particles) {
    std::atomic<int> totalNeighbors = 0;
    std::atomic<int> activeCount = 0;
    
    std::for_each(std::execution::par, particles.begin(), particles.end(),
        [&](FluidParticle& p) {
            if (!p.active) return;
            
            // Clear previous neighbors
            p.neighbors.clear();
            
            // Query spatial grid
            std::vector<int> nearbyParticles = m_SpatialGrid->QuerySphere(
                p.predictedPosition, m_KernelRadius);
            
            // Filter by actual distance
            for (int j : nearbyParticles) {
                if (&p == &particles[j]) continue;  // Skip self (pointer comparison)
                if (!particles[j].active) continue;
                
                Vec3 diff = particles[j].predictedPosition - p.predictedPosition;
                float distSq = diff.LengthSquared();
                
                if (distSq < m_KernelRadius * m_KernelRadius) {
                    p.neighbors.push_back(j);
                }
            }
            
            totalNeighbors += static_cast<int>(p.neighbors.size());
            activeCount++;
        }
    );
    
    m_Stats.avgNeighborsPerParticle = activeCount > 0 ? totalNeighbors / activeCount : 0;
}

void PBDFluidSolver::SolveConstraints(std::vector<FluidParticle>& particles,
                                     const std::vector<FluidType>& fluidTypes,
                                     int iterations) {
    for (int iter = 0; iter < iterations; ++iter) {
        // Compute density and pressure
        ComputeDensityAndPressure(particles, fluidTypes);
        
        // Compute lambda (Lagrange multipliers)
        ComputeLambda(particles, fluidTypes);
        
        // Compute position corrections
        ComputeDeltaPosition(particles, fluidTypes);
        
        // Apply position corrections
        for (auto& p : particles) {
            if (!p.active) continue;
            p.predictedPosition = p.predictedPosition + p.deltaPosition;
        }
    }
}

void PBDFluidSolver::ComputeDensityAndPressure(std::vector<FluidParticle>& particles,
                                              const std::vector<FluidType>& fluidTypes) {
    std::for_each(std::execution::par, particles.begin(), particles.end(),
        [&](FluidParticle& p) {
            if (!p.active) return;
            
            float density = 0.0f;
            const FluidType& type = fluidTypes[p.fluidType];
            
            // Self contribution
            density += type.particleMass * Poly6Kernel(0.0f, m_KernelRadius);
            
            // Neighbor contributions
            for (int neighborIdx : p.neighbors) {
                const auto& neighbor = particles[neighborIdx];
                Vec3 diff = p.predictedPosition - neighbor.predictedPosition;
                float dist = diff.Length();
                
                density += type.particleMass * Poly6Kernel(dist, m_KernelRadius);
            }
            
            p.density = density;
        }
    );
}

void PBDFluidSolver::ComputeLambda(std::vector<FluidParticle>& particles,
                                  const std::vector<FluidType>& fluidTypes) {
    std::for_each(std::execution::par, particles.begin(), particles.end(),
        [&](FluidParticle& p) {
            if (!p.active) return;
            
            const FluidType& type = fluidTypes[p.fluidType];
            float restDensity = type.restDensity;
            
            // Constraint: C = density / restDensity - 1
            float constraint = p.density / restDensity - 1.0f;
            
            // Compute gradient sum
            float gradientSum = 0.0f;
            Vec3 gradientI(0, 0, 0);
            
            for (int neighborIdx : p.neighbors) {
                const auto& neighbor = particles[neighborIdx];
                Vec3 diff = p.predictedPosition - neighbor.predictedPosition;
                
                Vec3 gradient = SpikyGradient(diff, m_KernelRadius) / restDensity;
                gradientI = gradientI + gradient;
                gradientSum += gradient.LengthSquared();
            }
            
            gradientSum += gradientI.LengthSquared();
            
            // Compute lambda with relaxation
            p.lambda = -constraint / (gradientSum + m_RelaxationEpsilon);
        }
    );
}

void PBDFluidSolver::ComputeDeltaPosition(std::vector<FluidParticle>& particles,
                                         const std::vector<FluidType>& fluidTypes) {
    std::for_each(std::execution::par, particles.begin(), particles.end(),
        [&](FluidParticle& p) {
            if (!p.active) return;
            
            const FluidType& type = fluidTypes[p.fluidType];
            float restDensity = type.restDensity;
            
            Vec3 deltaPos(0, 0, 0);
            
            for (int neighborIdx : p.neighbors) {
                const auto& neighbor = particles[neighborIdx];
                Vec3 diff = p.predictedPosition - neighbor.predictedPosition;
                
                Vec3 gradient = SpikyGradient(diff, m_KernelRadius);
                deltaPos = deltaPos + gradient * (p.lambda + neighbor.lambda) / restDensity;
            }
            
            p.deltaPosition = deltaPos;
        }
    );
}

void PBDFluidSolver::UpdateVelocities(std::vector<FluidParticle>& particles, float dt) {
    float invDt = 1.0f / dt;
    
    std::for_each(std::execution::par, particles.begin(), particles.end(),
        [&](FluidParticle& p) {
            if (!p.active) return;
            
            // Update velocity from position change
            p.velocity = (p.predictedPosition - p.position) * invDt;
            
            // Update position
            p.position = p.predictedPosition;
        }
    );
}

void PBDFluidSolver::ApplyViscosity(std::vector<FluidParticle>& particles,
                                   const std::vector<FluidType>& fluidTypes) {
    if (m_ViscosityScale <= 0.0f) return;
    
    for (auto& p : particles) {
        if (!p.active) continue;
        
        const FluidType& type = fluidTypes[p.fluidType];
        Vec3 viscosityForce(0, 0, 0);
        
        for (int neighborIdx : p.neighbors) {
            const auto& neighbor = particles[neighborIdx];
            Vec3 diff = p.position - neighbor.position;
            float dist = diff.Length();
            
            Vec3 velocityDiff = neighbor.velocity - p.velocity;
            viscosityForce = viscosityForce + velocityDiff * ViscosityLaplacian(dist, m_KernelRadius);
        }
        
        p.velocity = p.velocity + viscosityForce * type.viscosity * m_ViscosityScale;
    }
}

void PBDFluidSolver::ApplySurfaceTension(std::vector<FluidParticle>& particles,
                                        const std::vector<FluidType>& fluidTypes) {
    if (m_SurfaceTensionScale <= 0.0f) return;
    
    for (auto& p : particles) {
        if (!p.active) continue;
        
        const FluidType& type = fluidTypes[p.fluidType];
        Vec3 cohesionForce(0, 0, 0);
        
        for (int neighborIdx : p.neighbors) {
            const auto& neighbor = particles[neighborIdx];
            Vec3 diff = neighbor.position - p.position;
            float dist = diff.Length();
            
            if (dist > 0.0001f) {
                cohesionForce = cohesionForce + diff.Normalized() * type.surfaceTension;
            }
        }
        
        p.velocity = p.velocity + cohesionForce * m_SurfaceTensionScale * 0.01f;
    }
}

void PBDFluidSolver::HandleBoundaryCollisions(std::vector<FluidParticle>& particles) {
    const float damping = 0.5f;  // Velocity damping on collision
    
    for (auto& p : particles) {
        if (!p.active) continue;
        
        // Check each axis
        if (p.position.x < m_BoundaryMin.x) {
            p.position.x = m_BoundaryMin.x;
            p.velocity.x = -p.velocity.x * damping;
        }
        if (p.position.x > m_BoundaryMax.x) {
            p.position.x = m_BoundaryMax.x;
            p.velocity.x = -p.velocity.x * damping;
        }
        
        if (p.position.y < m_BoundaryMin.y) {
            p.position.y = m_BoundaryMin.y;
            p.velocity.y = -p.velocity.y * damping;
        }
        if (p.position.y > m_BoundaryMax.y) {
            p.position.y = m_BoundaryMax.y;
            p.velocity.y = -p.velocity.y * damping;
        }
        
        if (p.position.z < m_BoundaryMin.z) {
            p.position.z = m_BoundaryMin.z;
            p.velocity.z = -p.velocity.z * damping;
        }
        if (p.position.z > m_BoundaryMax.z) {
            p.position.z = m_BoundaryMax.z;
            p.velocity.z = -p.velocity.z * damping;
        }
    }
}

void PBDFluidSolver::HandleRigidBodyCollisions(std::vector<FluidParticle>& particles) {
    if (!m_PhysicsBackend) return;

    for (auto& p : particles) {
        if (!p.active) continue;

        Vec3 start = p.position;
        Vec3 end = p.predictedPosition;
        Vec3 dir = end - start;
        float dist = dir.Length();

        if (dist < 0.001f) continue;
        
        // Normalize direction
        dir = dir * (1.0f / dist);
        
        // Add a small margin to strict segment to detect very close walls
        float traceDist = dist + 0.01f; // Margin

        RaycastHit hit;
        
        // Raycast against physics scene (Rigid + Soft Bodies)
        if (m_PhysicsBackend->Raycast(start, start + dir * traceDist, hit)) {
            // If contact is within the movement step
            if (hit.distance <= dist) {
                // 1. Position Projection (Non-penetration)
                // Move particle to surface + epsilon
                Vec3 normal = hit.normal;
                p.predictedPosition = hit.position + normal * 0.001f;
                
                // 2. Velocity Response
                // Split velocity into normal and tangential components
                float vDotN = p.velocity.Dot(normal);
                
                // Only reflect if moving into the wall
                if (vDotN < 0) {
                    Vec3 vn = normal * vDotN;
                    Vec3 vt = p.velocity - vn;
                    
                    // Apply friction and restitution
                    float friction = 0.1f;    // Low friction for fluid
                    float restitution = 0.0f; // No bounce (inelastic)
                    
                    p.velocity = vt * (1.0f - friction) - vn * restitution;
                    
                    // TWO-WAY COUPLING: Apply impulse to Rigid Body
                    // Impulse J = mass * delta_v
                    // We approximate the impulse as the force needed to stop the particle's normal velocity
                    // or the force exerted by the position correction.
                    // Simple approach: The particle lost momentum 'mass * vn'. That momentum went into the wall.
                    // Since vn is negative (into wall), the impulse on wall is 'mass * vn' (in direction of normal? No).
                    // Particle initial momentum p_in = m * v_in. Final p_out = m * v_out.
                    // Change = p_out - p_in.
                    // Impulse on particle J_p = m * (v_out - v_in).
                    // Impulse on wall J_w = -J_p = m * (v_in - v_out).
                    
                    // v_out = vt * (1-f) - vn * r.
                    // v_in = vt + vn. (Decomposed).
                    // Let's focus on normal component for push.
                    // v_in_n = vn (negative). v_out_n = -vn * r (positive).
                    // J_w_n = m * (vn - (-vn*r)) = m * vn * (1+r).
                    // Since vn is towards wall (negative dot), J_w_n is negative (towards wall).
                    // Vector direction: normal is OUT of wall. so vn vector is INTO wall.
                    // The impulse vector should be INTO the wall.
                    
                    // Let's use the actual velocity change for accuracy including friction.
                    /*
                    Vec3 v_old = p.velocity; // This 'p.velocity' was just updated above? No, we updated it.
                    // Need old velocity.
                    // Let's use the computed response.
                    */
                    
                    if (hit.userData) {
                        // Re-calculate the change we just made or estimate it
                        // Impulse = mass * (velocity_before - velocity_after) ?? No, Force on body = Force FROM particle.
                        // Force on Particle = m * a.
                        // Force on Body = - Force on Particle.
                        
                        // Let's just apply a push based on penetration/impact.
                        // A robust way for PBD is to rely on the position correction.
                        // But velocity change is fine for impact.
                        
                        // Force magnitude roughly:
                        float impulseMag = std::abs(vDotN) * fluidTypes[p.fluidType].particleMass * (1.0f + restitution);
                        
                        // Direction: Into the wall (opposite to normal)
                        Vec3 impulseDir = -normal;
                        
                        // Apply scaled impulse
                        float interactionScale = 1.0f; // Tune this to control fluid strength
                        m_PhysicsBackend->ApplyImpulse(hit.userData, impulseDir * impulseMag * interactionScale, hit.point);
                    }
                }
            } else {
                // Moving towards wall but haven't hit it yet in this step?
                // For PBD, we usually only care if we penetrate or cross.
                // The raycast covers the path (CCD). 
                // If hit distance > dist, we didn't reach it yet. Safe.
            }
        }
    }
}

// SPH Kernel Functions

float PBDFluidSolver::Poly6Kernel(float r, float h) const {
    if (r >= h) return 0.0f;
    
    float h2 = h * h;
    float h9 = h2 * h2 * h2 * h2 * h;
    float diff = h2 - r * r;
    
    return 315.0f / (64.0f * static_cast<float>(M_PI) * h9) * diff * diff * diff;
}

Vec3 PBDFluidSolver::SpikyGradient(const Vec3& r, float h) const {
    float dist = r.Length();
    if (dist >= h || dist < 0.0001f) return Vec3(0, 0, 0);
    
    float h6 = h * h * h * h * h * h;
    float diff = h - dist;
    float coeff = -45.0f / (static_cast<float>(M_PI) * h6) * diff * diff;
    
    return r.Normalized() * coeff;
}

float PBDFluidSolver::ViscosityLaplacian(float r, float h) const {
    if (r >= h) return 0.0f;
    
    float h6 = h * h * h * h * h * h;
    return 45.0f / (static_cast<float>(M_PI) * h6) * (h - r);
}
