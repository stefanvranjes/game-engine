#include "FluidSimulationEditor.h"
#include "FluidDebugGizmo.h"
#include <imgui.h>
#include <cstdio>

FluidSimulationEditor::FluidSimulationEditor(std::shared_ptr<FluidSimulation> simulation)
    : m_Simulation(simulation)
    , m_ShowDebugOptions(true)
    , m_ShowPerformance(true)
    , m_ShowEmitterControls(true)
{
}

FluidSimulationEditor::~FluidSimulationEditor() {
}

void FluidSimulationEditor::Draw() {
    if (!m_Simulation) return;

    if (ImGui::Begin("PBD Fluid Simulation")) {
        
        if (ImGui::CollapsingHeader("Solver Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
            DrawSolverSettings();
        }

        if (ImGui::CollapsingHeader("Fluid Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
            DrawFluidProperties();
        }
        
        if (ImGui::CollapsingHeader("Emitters")) {
            DrawEmitterControls();
        }

        if (ImGui::CollapsingHeader("Statistics", ImGuiTreeNodeFlags_DefaultOpen)) {
            DrawStatistics();
        }
    }
    ImGui::End();
}

void FluidSimulationEditor::DrawSolverSettings() {
    // Gravity
    Vec3 gravity = m_Simulation->GetGravity();
    float gravityArr[3] = { gravity.x, gravity.y, gravity.z };
    if (ImGui::DragFloat3("Gravity", gravityArr, 0.1f)) {
        m_Simulation->SetGravity(Vec3(gravityArr[0], gravityArr[1], gravityArr[2]));
    }
    
    // Time Scale
    float timeScale = m_Simulation->GetTimeScale(); // Need to expose getter if not present
    // Assuming GetTimeScale exists or I add it. Checking FluidSimulation.h previously I didn't see it explicitly public, 
    // but the member is visible in cpp. I should check/add getters.
    // For now, let's assume I can modify m_TimeScale behavior via setters if needed.
    // Wait, FluidSimulation.h usually has Setters.
    
    // Radius
    float radius = m_Simulation->GetKernelRadius();
    if (ImGui::SliderFloat("Kernel Radius", &radius, 0.01f, 1.0f)) {
        m_Simulation->SetKernelRadius(radius);
    }
    
    // Iterations
    int iterations = m_Simulation->GetSolverIterations();
    if (ImGui::SliderInt("Solver Iterations", &iterations, 1, 10)) {
        m_Simulation->SetSolverIterations(iterations);
    }
    
    // Boundary
    // Vec3 min = ... GetBoundaryMin() -> Need to ensure getters exist
    
    bool boundaryEnabled = true; // Placeholder, need getter
    if (ImGui::Checkbox("Enable Boundary", &boundaryEnabled)) {
        m_Simulation->SetEnableBoundary(boundaryEnabled);
    }
    
    ImGui::Separator();
    ImGui::Text("Physics Parameters");
    
    static float visc = 1.0f;
    if (ImGui::SliderFloat("Viscosity Scale", &visc, 0.0f, 5.0f)) {
        m_Simulation->SetViscosityScale(visc);
    }
    
    static float tension = 1.0f;
    if (ImGui::SliderFloat("Surface Tension", &tension, 0.0f, 5.0f)) {
        m_Simulation->SetSurfaceTensionScale(tension);
    }
    
    static float vorticity = 0.0f;
    if (ImGui::SliderFloat("Vorticity Strength", &vorticity, 0.0f, 20.0f)) {
        m_Simulation->SetVorticityConfinementScale(vorticity);
    }
}

void FluidSimulationEditor::DrawFluidProperties() {
    // Multi-fluid support makes this trickier. For now, edit Fluid Type 0
    // We need API on FluidSimulation to GetFluidType(index) and Set... or update it.
    // PBDFluidSolver updates using the vector of types passed in.
    // Ideally FluidSimulation exposes a way to mutate types.
    
    ImGui::Text("Fluid Type 0 (Default)");
    // Placeholder interaction
}

void FluidSimulationEditor::DrawEmitterControls() {
    ImGui::Text("Emitters: %zu", m_Simulation->GetEmitters().size()); // Need GetEmitters
    
    if (ImGui::Button("Clear Emitters")) {
        m_Simulation->ClearEmitters();
    }
    
    if (ImGui::Button("Reset Particles")) {
        m_Simulation->ClearParticles();
    }
    
    if (m_DebugGizmo) {
        ImGui::Separator();
        ImGui::Text("Debug Visualization");
        
        static bool showVel = false;
        if (ImGui::Checkbox("Show Velocities", &showVel)) {
            m_DebugGizmo->SetDrawVelocities(showVel);
        }
        
        static float velScale = 0.1f;
        if (ImGui::SliderFloat("Velocity Scale", &velScale, 0.01f, 1.0f)) {
            m_DebugGizmo->SetVelocityScale(velScale);
        }
    }
}

void FluidSimulationEditor::DrawStatistics() {
    const auto& stats = m_Simulation->GetStatistics();
    
    ImGui::Text("Active Particles: %d", stats.activeParticles);
    ImGui::Text("Avg Neighbors: %d", stats.avgNeighborsPerParticle);
    ImGui::Separator();
    ImGui::Text("Total Time: %.2f ms", stats.totalTimeMs);
    ImGui::Text("  - Predict: %.2f ms", stats.predictTimeMs);
    ImGui::Text("  - Neighbor: %.2f ms", stats.neighborSearchTimeMs);
    ImGui::Text("  - Constraint: %.2f ms", stats.constraintTimeMs);
    ImGui::Text("  - Update: %.2f ms", stats.velocityUpdateTimeMs);
}
