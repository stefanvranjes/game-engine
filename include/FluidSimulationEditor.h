#pragma once

#include "FluidSimulation.h"
#include <memory>

/**
 * @brief Editor UI panel for Fluid Simulation configuration
 */
class FluidSimulationEditor {
public:
    FluidSimulationEditor(std::shared_ptr<FluidSimulation> simulation);
    ~FluidSimulationEditor();

    /**
     * @brief Draw the editor panel using ImGui
     */
    void Draw();

private:
    std::shared_ptr<FluidSimulation> m_Simulation;
    
    // UI State
    bool m_ShowDebugOptions;
    bool m_ShowPerformance;
    bool m_ShowEmitterControls;
    
    // Helper methods for sections
    void DrawSolverSettings();
    void DrawFluidProperties();
    void DrawRenderingOptions(); // Placeholder if renderer is exposed later
    void DrawEmitterControls();
    void DrawStatistics();
    
    // Debug Gizmo
    std::shared_ptr<class FluidDebugGizmo> m_DebugGizmo;
    
public:
    void SetDebugGizmo(std::shared_ptr<FluidDebugGizmo> gizmo) { m_DebugGizmo = gizmo; }
};
