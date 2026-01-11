#pragma once

#include "Gizmo.h"
#include <memory>
#include <vector>

class FluidSimulation;

/**
 * @brief Gizmo for visualizing fluid simulation debug data
 */
class FluidDebugGizmo : public Gizmo {
public:
    FluidDebugGizmo(std::shared_ptr<FluidSimulation> simulation);
    virtual ~FluidDebugGizmo();

    // Gizmo overrides
    void Draw(Shader* shader, const Camera& camera) override;
    
    // Interaction overrides (No interaction for debug visualization)
    bool OnMousePress(const Ray& ray) override { return false; }
    void OnMouseRelease() override {}
    void OnMouseDrag(const Ray& ray, const Camera& camera) override {}
    bool OnMouseMove(const Ray& ray) override { return false; }

    // Configuration
    void SetDrawVelocities(bool enabled) { m_DrawVelocities = enabled; }
    void SetDrawSpatialGrid(bool enabled) { m_DrawSpatialGrid = enabled; }
    void SetVelocityScale(float scale) { m_VelocityScale = scale; }
    
private:
    std::shared_ptr<FluidSimulation> m_Simulation;
    
    bool m_DrawVelocities = false;
    bool m_DrawSpatialGrid = false;
    float m_VelocityScale = 1.0f;
    
    // Rendering resources
    unsigned int m_LineVAO = 0;
    unsigned int m_LineVBO = 0;
    
    void RenderLines(const std::vector<Vec3>& points, const std::vector<Vec3>& colors);
};
