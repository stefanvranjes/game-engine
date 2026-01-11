#pragma once

#include "FluidParticle.h"
#include "FluidType.h"
#include "SpatialHashGrid.h"
#include "Math/Vec3.h"
#include <vector>
#include <memory>

class PhysXBackend;

/**
 * @brief Position-Based Dynamics fluid solver
 * 
 * Implements PBD-based fluid simulation with:
 * - Density constraints for incompressibility
 * - Viscosity for realistic fluid motion
 * - Surface tension for cohesion
 * - Boundary collision handling
 */
class PBDFluidSolver {
public:
    PBDFluidSolver();
    virtual ~PBDFluidSolver();
    
    /**
     * @brief Initialize the solver
     * @param backend Physics backend for collision queries
     */
    virtual void Initialize(PhysXBackend* backend);
    
    /**
     * @brief Update simulation for one timestep
     * @param particles Particle array to simulate
     * @param fluidTypes Fluid type definitions
     * @param deltaTime Time step
     */
    virtual void Update(std::vector<FluidParticle>& particles, 
                const std::vector<FluidType>& fluidTypes,
                float deltaTime);
    
    /**
     * @brief Set solver parameters
     */
    void SetKernelRadius(float radius) { m_KernelRadius = radius; }
    void SetSolverIterations(int iterations) { m_SolverIterations = iterations; }
    void SetGravity(const Vec3& gravity) { m_Gravity = gravity; }
    void SetBoundaryMin(const Vec3& min) { m_BoundaryMin = min; }
    void SetBoundaryMax(const Vec3& max) { m_BoundaryMax = max; }
    void SetEnableBoundary(bool enable) { m_EnableBoundary = enable; }
    void SetRelaxationParameter(float epsilon) { m_RelaxationEpsilon = epsilon; }
    void SetViscosityScale(float scale) { m_ViscosityScale = scale; }
    void SetSurfaceTensionScale(float scale) { m_SurfaceTensionScale = scale; }
    
    /**
     * @brief Get solver parameters
     */
    float GetKernelRadius() const { return m_KernelRadius; }
    int GetSolverIterations() const { return m_SolverIterations; }
    Vec3 GetGravity() const { return m_Gravity; }
    Vec3 GetBoundaryMin() const { return m_BoundaryMin; }
    Vec3 GetBoundaryMax() const { return m_BoundaryMax; }
    bool IsBoundaryEnabled() const { return m_EnableBoundary; }
    
    /**
     * @brief Get performance statistics
     */
    struct Statistics {
        float predictTimeMs;
        float neighborSearchTimeMs;
        float densityTimeMs;
        float constraintTimeMs;
        float velocityUpdateTimeMs;
        float totalTimeMs;
        int activeParticles;
        int avgNeighborsPerParticle;
    };
    const Statistics& GetStatistics() const { return m_Stats; }

protected:
    // Solver parameters
    float m_KernelRadius;
    int m_SolverIterations;
    Vec3 m_Gravity;
    Vec3 m_BoundaryMin;
    Vec3 m_BoundaryMax;
    bool m_EnableBoundary;
    float m_RelaxationEpsilon;  // CFM relaxation parameter
    float m_ViscosityScale;
    float m_SurfaceTensionScale;
    
    // Spatial acceleration
    std::unique_ptr<SpatialHashGrid> m_SpatialGrid;
    
    // Physics backend
    PhysXBackend* m_PhysicsBackend;
    
    // Statistics
    Statistics m_Stats;
    
    // Simulation steps
    void ApplyExternalForces(std::vector<FluidParticle>& particles, 
                            const std::vector<FluidType>& fluidTypes,
                            float dt);
    
    void PredictPositions(std::vector<FluidParticle>& particles, float dt);
    
    void BuildSpatialGrid(const std::vector<FluidParticle>& particles);
    
    void FindNeighbors(std::vector<FluidParticle>& particles);
    
    void SolveConstraints(std::vector<FluidParticle>& particles,
                         const std::vector<FluidType>& fluidTypes,
                         int iterations);
    
    void UpdateVelocities(std::vector<FluidParticle>& particles, float dt);
    
    void ApplyViscosity(std::vector<FluidParticle>& particles,
                       const std::vector<FluidType>& fluidTypes);
    
    void ApplySurfaceTension(std::vector<FluidParticle>& particles,
                            const std::vector<FluidType>& fluidTypes);
    
    void HandleBoundaryCollisions(std::vector<FluidParticle>& particles);
    
    void HandleRigidBodyCollisions(std::vector<FluidParticle>& particles);
    
    // SPH kernel functions
    float Poly6Kernel(float r, float h) const;
    Vec3 SpikyGradient(const Vec3& r, float h) const;
    float ViscosityLaplacian(float r, float h) const;
    
    // Density constraint helpers
    void ComputeDensityAndPressure(std::vector<FluidParticle>& particles,
                                   const std::vector<FluidType>& fluidTypes);
    
    void ComputeLambda(std::vector<FluidParticle>& particles,
                      const std::vector<FluidType>& fluidTypes);
    
    void ComputeDeltaPosition(std::vector<FluidParticle>& particles,
                             const std::vector<FluidType>& fluidTypes);
};
