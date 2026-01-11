#include "FluidSimulation.h"
#include "PhysXBackend.h"
#include <algorithm>

FluidSimulation::FluidSimulation()
    : m_MaxParticles(50000)
    , m_TimeScale(1.0f)
    , m_Substeps(1)
    , m_PhysicsBackend(nullptr)
    , m_Solver(std::make_unique<PBDFluidSolver>())
{
    // Add default water fluid type
    m_FluidTypes.push_back(FluidType::Water());
}

FluidSimulation::~FluidSimulation() {
}

void FluidSimulation::Initialize(PhysXBackend* backend) {
    m_PhysicsBackend = backend;
    m_Solver->Initialize(backend);
}

void FluidSimulation::Update(float deltaTime) {
    // Apply time scale
    deltaTime *= m_TimeScale;
    
    // Update emitters
    UpdateEmitters(deltaTime);
    
    // Update particle lifetimes
    UpdateParticleLifetimes(deltaTime);
    
    // Remove dead particles
    RemoveDeadParticles();
    
    // Enforce particle limit
    EnforceParticleLimit();
    
    // Simulate with substeps
    float substepDt = deltaTime / static_cast<float>(m_Substeps);
    for (int i = 0; i < m_Substeps; ++i) {
        m_Solver->Update(m_Particles, m_FluidTypes, substepDt);
    }
}

void FluidSimulation::AddParticle(const Vec3& position, const Vec3& velocity, int fluidType) {
    if (static_cast<int>(m_Particles.size()) >= m_MaxParticles) {
        return;  // Particle limit reached
    }
    
    FluidParticle particle(position, velocity, fluidType);
    m_Particles.push_back(particle);
}

void FluidSimulation::AddEmitter(std::shared_ptr<FluidEmitter> emitter) {
    m_Emitters.push_back(emitter);
}

void FluidSimulation::RemoveEmitter(std::shared_ptr<FluidEmitter> emitter) {
    auto it = std::find(m_Emitters.begin(), m_Emitters.end(), emitter);
    if (it != m_Emitters.end()) {
        m_Emitters.erase(it);
    }
}

void FluidSimulation::ClearEmitters() {
    m_Emitters.clear();
}

int FluidSimulation::AddFluidType(const FluidType& fluidType) {
    m_FluidTypes.push_back(fluidType);
    return static_cast<int>(m_FluidTypes.size()) - 1;
}

void FluidSimulation::ClearParticles() {
    m_Particles.clear();
}

void FluidSimulation::SetGravity(const Vec3& gravity) {
    m_Solver->SetGravity(gravity);
}

void FluidSimulation::SetKernelRadius(float radius) {
    m_Solver->SetKernelRadius(radius);
}

void FluidSimulation::SetSolverIterations(int iterations) {
    m_Solver->SetSolverIterations(iterations);
}

void FluidSimulation::SetBoundaryMin(const Vec3& min) {
    m_Solver->SetBoundaryMin(min);
}

void FluidSimulation::SetBoundaryMax(const Vec3& max) {
    m_Solver->SetBoundaryMax(max);
}

void FluidSimulation::SetEnableBoundary(bool enable) {
    m_Solver->SetEnableBoundary(enable);
}

Vec3 FluidSimulation::GetGravity() const {
    return m_Solver->GetGravity();
}

float FluidSimulation::GetKernelRadius() const {
    return m_Solver->GetKernelRadius();
}

int FluidSimulation::GetSolverIterations() const {
    return m_Solver->GetSolverIterations();
}

int FluidSimulation::GetActiveParticleCount() const {
    int count = 0;
    for (const auto& p : m_Particles) {
        if (p.active) count++;
    }
    return count;
}

const PBDFluidSolver::Statistics& FluidSimulation::GetStatistics() const {
    return m_Solver->GetStatistics();
}

void FluidSimulation::UpdateEmitters(float deltaTime) {
    for (auto& emitter : m_Emitters) {
        emitter->Update(deltaTime, m_Particles);
    }
}

void FluidSimulation::UpdateParticleLifetimes(float deltaTime) {
    for (auto& p : m_Particles) {
        if (!p.active) continue;
        
        p.age += deltaTime;
        
        // Check if particle has exceeded lifetime
        if (p.lifetime > 0.0f && p.age >= p.lifetime) {
            p.active = false;
        }
    }
}

void FluidSimulation::RemoveDeadParticles() {
    // Remove inactive particles
    m_Particles.erase(
        std::remove_if(m_Particles.begin(), m_Particles.end(),
            [](const FluidParticle& p) { return !p.active; }),
        m_Particles.end()
    );
}

void FluidSimulation::EnforceParticleLimit() {
    // If we exceed the limit, remove oldest particles
    while (static_cast<int>(m_Particles.size()) > m_MaxParticles) {
        // Find oldest particle
        auto oldestIt = m_Particles.begin();
        float maxAge = oldestIt->age;
        
        for (auto it = m_Particles.begin(); it != m_Particles.end(); ++it) {
            if (it->age > maxAge) {
                maxAge = it->age;
                oldestIt = it;
            }
        }
        
        // Remove oldest particle
        m_Particles.erase(oldestIt);
    }
}
