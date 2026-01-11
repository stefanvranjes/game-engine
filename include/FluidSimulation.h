#pragma once

#include "FluidParticle.h"
#include "FluidType.h"
#include "FluidEmitter.h"
#include "PBDFluidSolver.h"
#include "FoamParticleSystem.h"
#include "Math/Vec3.h"
#include <vector>
#include <memory>

class PhysXBackend;

/**
 * @brief Main fluid simulation component
 * 
 * Manages fluid particles, emitters, and simulation using PBD solver
 */
class FluidSimulation {
public:
    FluidSimulation();
    ~FluidSimulation();
    
    /**
     * @brief Initialize the simulation
     * @param backend Physics backend for collision queries
     */
    void Initialize(PhysXBackend* backend);
    
    /**
     * @brief Update simulation for one timestep
     * @param deltaTime Time step
     */
    void Update(float deltaTime);
    
    /**
     * @brief Add a particle to the simulation
     * @param position Initial position
     * @param velocity Initial velocity
     * @param fluidType Fluid type index
     */
    void AddParticle(const Vec3& position, const Vec3& velocity, int fluidType = 0);
    
    /**
     * @brief Add an emitter
     * @param emitter Emitter to add
     */
    void AddEmitter(std::shared_ptr<FluidEmitter> emitter);
    
    /**
     * @brief Remove an emitter
     * @param emitter Emitter to remove
     */
    void RemoveEmitter(std::shared_ptr<FluidEmitter> emitter);
    
    /**
     * @brief Clear all emitters
     */
    void ClearEmitters();
    
    /**
     * @brief Add a fluid type
     * @param fluidType Fluid type to add
     * @return Index of the added fluid type
     */
    int AddFluidType(const FluidType& fluidType);
    
    /**
     * @brief Get fluid type by index
     */
    const FluidType& GetFluidType(int index) const { return m_FluidTypes[index]; }
    FluidType& GetFluidType(int index) { return m_FluidTypes[index]; }
    
    /**
     * @brief Get all particles
     */
    const std::vector<FluidParticle>& GetParticles() const { return m_Particles; }
    std::vector<FluidParticle>& GetParticles() { return m_Particles; }
    
    /**
     * @brief Get all emitters
     */
    const std::vector<std::shared_ptr<FluidEmitter>>& GetEmitters() const { return m_Emitters; }
    
    /**
     * @brief Clear all particles
     */
    void ClearParticles();
    
    /**
     * @brief Set simulation parameters
     */
    void SetGravity(const Vec3& gravity);
    void SetKernelRadius(float radius);
    void SetSolverIterations(int iterations);
    void SetBoundaryMin(const Vec3& min);
    void SetBoundaryMax(const Vec3& max);
    void SetEnableBoundary(bool enable);
    void SetMaxParticles(int maxParticles) { m_MaxParticles = maxParticles; }
    void SetTimeScale(float scale) { m_TimeScale = scale; }
    void SetSubsteps(int substeps) { m_Substeps = substeps; }
    
    /**
     * @brief Get simulation parameters
     */
    Vec3 GetGravity() const;
    float GetKernelRadius() const;
    int GetSolverIterations() const;
    int GetActiveParticleCount() const;
    int GetMaxParticles() const { return m_MaxParticles; }
    float GetTimeScale() const { return m_TimeScale; }
    int GetSubsteps() const { return m_Substeps; }
    
    /**
     * @brief Get solver statistics
     */
    const PBDFluidSolver::Statistics& GetStatistics() const;
    
    /**
     * @brief Get solver
     */
    PBDFluidSolver* GetSolver() { return m_Solver.get(); }
    const PBDFluidSolver* GetSolver() const { return m_Solver.get(); }
    
    /**
     * @brief Foam particle system
     */
    FoamParticleSystem* GetFoamSystem() { return m_FoamSystem.get(); }
    const FoamParticleSystem* GetFoamSystem() const { return m_FoamSystem.get(); }
    
    /**
     * @brief Foam generation parameters
     */
    void SetFoamEnabled(bool enabled) { m_FoamEnabled = enabled; }
    bool IsFoamEnabled() const { return m_FoamEnabled; }
    void SetFoamVelocityThreshold(float threshold) { m_FoamVelocityThreshold = threshold; }
    float GetFoamVelocityThreshold() const { return m_FoamVelocityThreshold; }
    void SetFoamSpawnRate(float rate) { m_FoamSpawnRate = rate; }
    float GetFoamSpawnRate() const { return m_FoamSpawnRate; }
    void SetFoamMergeRadius(float radius);
    void SetFoamAdhesion(float strength);

private:
    // Particles and fluid types
    std::vector<FluidParticle> m_Particles;
    std::vector<FluidType> m_FluidTypes;
    
    // Emitters
    std::vector<std::shared_ptr<FluidEmitter>> m_Emitters;
    
    // Solver
    std::unique_ptr<PBDFluidSolver> m_Solver;
    
    // Foam particle system
    std::unique_ptr<FoamParticleSystem> m_FoamSystem;
    bool m_FoamEnabled;
    float m_FoamVelocityThreshold;
    float m_FoamSpawnRate;
    
    // Simulation parameters
    int m_MaxParticles;
    float m_TimeScale;
    int m_Substeps;
    
    // Physics backend
    PhysXBackend* m_PhysicsBackend;
    
    // Helper methods
    void UpdateEmitters(float deltaTime);
    void UpdateParticleLifetimes(float deltaTime);
    void RemoveDeadParticles();
    void EnforceParticleLimit();
};
