#include "Application.h"
#include "FluidSimulation.h"
#include "FluidRenderer.h"
#include "FluidEmitter.h"
#include "PhysXBackend.h"
#include "Camera.h"
#include "ImGuiManager.h"
#include <memory>

/**
 * @brief Fluid simulation example demonstrating PBD fluid simulation
 * 
 * Features:
 * - Basic fluid simulation with gravity
 * - Configurable emitters
 * - Screen-space fluid rendering
 * - Interactive parameter tuning
 */
class FluidSimulationExample : public Application {
public:
    FluidSimulationExample() : Application("Fluid Simulation Example", 1920, 1080) {}
    
    void OnInit() override {
        // Initialize physics backend
        m_PhysicsBackend = std::make_unique<PhysXBackend>();
        m_PhysicsBackend->Initialize(Vec3(0, -9.81f, 0));
        
        // Initialize fluid simulation
        m_FluidSimulation = std::make_unique<FluidSimulation>();
        m_FluidSimulation->Initialize(m_PhysicsBackend.get());
        
        // Set simulation parameters
        m_FluidSimulation->SetKernelRadius(0.1f);
        m_FluidSimulation->SetSolverIterations(3);
        m_FluidSimulation->SetBoundaryMin(Vec3(-5, 0, -5));
        m_FluidSimulation->SetBoundaryMax(Vec3(5, 10, 5));
        m_FluidSimulation->SetEnableBoundary(true);
        m_FluidSimulation->SetMaxParticles(10000);
        m_FluidSimulation->SetSubsteps(2);
        
        // Add fluid types
        m_FluidSimulation->AddFluidType(FluidType::Water());
        m_FluidSimulation->AddFluidType(FluidType::Oil());
        
        // Create emitters
        auto waterEmitter = std::make_shared<FluidEmitter>();
        waterEmitter->SetPosition(Vec3(0, 5, 0));
        waterEmitter->SetVelocity(Vec3(0, -1, 0));
        waterEmitter->SetEmissionRate(100.0f);
        waterEmitter->SetFluidType(0);  // Water
        waterEmitter->SetEmissionShape(FluidEmitter::EmissionShape::Sphere);
        waterEmitter->SetSphereRadius(0.2f);
        waterEmitter->SetVelocitySpread(0.5f);
        m_FluidSimulation->AddEmitter(waterEmitter);
        
        auto oilEmitter = std::make_shared<FluidEmitter>();
        oilEmitter->SetPosition(Vec3(2, 5, 0));
        oilEmitter->SetVelocity(Vec3(0, -1, 0));
        oilEmitter->SetEmissionRate(50.0f);
        oilEmitter->SetFluidType(1);  // Oil
        oilEmitter->SetEmissionShape(FluidEmitter::EmissionShape::Point);
        oilEmitter->SetEnabled(false);  // Start disabled
        m_FluidSimulation->AddEmitter(oilEmitter);
        
        m_WaterEmitter = waterEmitter;
        m_OilEmitter = oilEmitter;
        
        // Initialize fluid renderer
        m_FluidRenderer = std::make_unique<FluidRenderer>();
        m_FluidRenderer->Initialize(GetWindowWidth(), GetWindowHeight());
        m_FluidRenderer->SetRenderMode(FluidRenderer::RenderMode::Particles);
        m_FluidRenderer->SetParticleSize(0.05f);
        
        // Setup camera
        m_Camera = std::make_unique<Camera>();
        m_Camera->SetPosition(Vec3(0, 5, 10));
        m_Camera->LookAt(Vec3(0, 3, 0));
        m_Camera->SetPerspective(45.0f, GetAspectRatio(), 0.1f, 100.0f);
    }
    
    void OnUpdate(float deltaTime) override {
        // Update camera
        UpdateCamera(deltaTime);
        
        // Update fluid simulation
        m_FluidSimulation->Update(deltaTime);
        
        // Update physics
        m_PhysicsBackend->Update(deltaTime);
    }
    
    void OnRender() override {
        // Clear screen
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Render fluid
        m_FluidRenderer->Render(m_FluidSimulation.get(), m_Camera.get());
        
        // Render UI
        RenderUI();
    }
    
    void OnShutdown() override {
        m_FluidRenderer->Shutdown();
        m_FluidSimulation.reset();
        m_PhysicsBackend->Shutdown();
    }
    
private:
    std::unique_ptr<PhysXBackend> m_PhysicsBackend;
    std::unique_ptr<FluidSimulation> m_FluidSimulation;
    std::unique_ptr<FluidRenderer> m_FluidRenderer;
    std::unique_ptr<Camera> m_Camera;
    
    std::shared_ptr<FluidEmitter> m_WaterEmitter;
    std::shared_ptr<FluidEmitter> m_OilEmitter;
    
    void UpdateCamera(float deltaTime) {
        // Simple orbit camera
        static float angle = 0.0f;
        angle += deltaTime * 0.2f;
        
        float radius = 10.0f;
        Vec3 pos(std::sin(angle) * radius, 5.0f, std::cos(angle) * radius);
        m_Camera->SetPosition(pos);
        m_Camera->LookAt(Vec3(0, 3, 0));
    }
    
    void RenderUI() {
        ImGui::Begin("Fluid Simulation");
        
        // Statistics
        ImGui::Text("Particles: %d / %d", 
                   m_FluidSimulation->GetActiveParticleCount(),
                   m_FluidSimulation->GetMaxParticles());
        
        auto stats = m_FluidSimulation->GetStatistics();
        ImGui::Text("Simulation Time: %.2f ms", stats.totalTimeMs);
        ImGui::Text("Avg Neighbors: %d", stats.avgNeighborsPerParticle);
        
        ImGui::Separator();
        
        // Simulation parameters
        if (ImGui::CollapsingHeader("Simulation Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
            float kernelRadius = m_FluidSimulation->GetKernelRadius();
            if (ImGui::SliderFloat("Kernel Radius", &kernelRadius, 0.05f, 0.5f)) {
                m_FluidSimulation->SetKernelRadius(kernelRadius);
            }
            
            int iterations = m_FluidSimulation->GetSolverIterations();
            if (ImGui::SliderInt("Solver Iterations", &iterations, 1, 10)) {
                m_FluidSimulation->SetSolverIterations(iterations);
            }
            
            int substeps = m_FluidSimulation->GetSubsteps();
            if (ImGui::SliderInt("Substeps", &substeps, 1, 5)) {
                m_FluidSimulation->SetSubsteps(substeps);
            }
            
            float timeScale = m_FluidSimulation->GetTimeScale();
            if (ImGui::SliderFloat("Time Scale", &timeScale, 0.0f, 2.0f)) {
                m_FluidSimulation->SetTimeScale(timeScale);
            }
        }
        
        // Emitters
        if (ImGui::CollapsingHeader("Emitters")) {
            bool waterEnabled = m_WaterEmitter->IsEnabled();
            if (ImGui::Checkbox("Water Emitter", &waterEnabled)) {
                m_WaterEmitter->SetEnabled(waterEnabled);
            }
            
            if (waterEnabled) {
                float waterRate = m_WaterEmitter->GetEmissionRate();
                if (ImGui::SliderFloat("Water Rate", &waterRate, 0.0f, 500.0f)) {
                    m_WaterEmitter->SetEmissionRate(waterRate);
                }
            }
            
            bool oilEnabled = m_OilEmitter->IsEnabled();
            if (ImGui::Checkbox("Oil Emitter", &oilEnabled)) {
                m_OilEmitter->SetEnabled(oilEnabled);
            }
            
            if (oilEnabled) {
                float oilRate = m_OilEmitter->GetEmissionRate();
                if (ImGui::SliderFloat("Oil Rate", &oilRate, 0.0f, 500.0f)) {
                    m_OilEmitter->SetEmissionRate(oilRate);
                }
            }
        }
        
        // Rendering
        if (ImGui::CollapsingHeader("Rendering")) {
            const char* renderModes[] = { "Particles", "Screen Space" };
            int currentMode = static_cast<int>(m_FluidRenderer->GetRenderMode());
            if (ImGui::Combo("Render Mode", &currentMode, renderModes, 2)) {
                m_FluidRenderer->SetRenderMode(static_cast<FluidRenderer::RenderMode>(currentMode));
            }
            
            float particleSize = m_FluidRenderer->GetParticleSize();
            if (ImGui::SliderFloat("Particle Size", &particleSize, 0.01f, 0.2f)) {
                m_FluidRenderer->SetParticleSize(particleSize);
            }
        }
        
        // Actions
        if (ImGui::Button("Clear Particles")) {
            m_FluidSimulation->ClearParticles();
        }
        
        ImGui::End();
    }
};

int main() {
    FluidSimulationExample app;
    app.Run();
    return 0;
}
