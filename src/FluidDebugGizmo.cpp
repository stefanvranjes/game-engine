#include "FluidDebugGizmo.h"
#include "FluidSimulation.h"
#include "FluidParticle.h"
#include <glad/glad.h>
#include <vector>

FluidDebugGizmo::FluidDebugGizmo(std::shared_ptr<FluidSimulation> simulation)
    : m_Simulation(simulation)
{
    glGenVertexArrays(1, &m_LineVAO);
    glGenBuffers(1, &m_LineVBO);
    
    glBindVertexArray(m_LineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_LineVBO);
    
    // Position (3 floats) + Color (3 floats)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    
    glBindVertexArray(0);
}

FluidDebugGizmo::~FluidDebugGizmo() {
    if (m_LineVAO) glDeleteVertexArrays(1, &m_LineVAO);
    if (m_LineVBO) glDeleteBuffers(1, &m_LineVBO);
}

void FluidDebugGizmo::Draw(Shader* shader, const Camera& camera) {
    if (!m_Simulation || !m_Enabled) return;
    
    std::vector<Vec3> points;
    std::vector<Vec3> colors;
    
    // Collect Velocity Vectors
    if (m_DrawVelocities) {
        const auto& particles = m_Simulation->GetParticles();
        for (const auto& p : particles) {
            if (!p.active) continue;
            
            Vec3 start = p.position;
            Vec3 end = p.position + p.velocity * m_VelocityScale; // Scale for visibility
            
            // Color by speed
            float speed = p.velocity.Length();
            Vec3 color(0.0f, 1.0f, 0.0f);
            if (speed > 5.0f) color = Vec3(1.0f, 1.0f, 0.0f);
            if (speed > 10.0f) color = Vec3(1.0f, 0.0f, 0.0f);
            
            points.push_back(start);
            points.push_back(end);
            
            colors.push_back(color);
            colors.push_back(color);
        }
    }
    
    // Collect Grid Lines (Placeholder - requires SpatialHashGrid exposure)
    if (m_DrawSpatialGrid) {
        // ... (Would need to iterate grid cells)
    }
    
    if (!points.empty()) {
        RenderLines(points, colors);
    }
}

void FluidDebugGizmo::RenderLines(const std::vector<Vec3>& points, const std::vector<Vec3>& colors) {
    if (points.empty()) return;
    
    std::vector<float> vertexData;
    vertexData.reserve(points.size() * 6);
    
    for (size_t i = 0; i < points.size(); ++i) {
        vertexData.push_back(points[i].x);
        vertexData.push_back(points[i].y);
        vertexData.push_back(points[i].z);
        
        vertexData.push_back(colors[i].x);
        vertexData.push_back(colors[i].y);
        vertexData.push_back(colors[i].z);
    }
    
    glBindVertexArray(m_LineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_LineVBO);
    glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_DYNAMIC_DRAW);
    
    glDrawArrays(GL_LINES, 0, points.size());
    
    glBindVertexArray(0);
}
