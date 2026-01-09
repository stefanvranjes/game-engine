#include "VertexHighlighter.h"
#include "PhysXSoftBody.h"
#include <algorithm>

// OpenGL includes (adjust based on your setup)
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/gl.h>

VertexHighlighter::VertexHighlighter()
    : m_HoveredVertex(-1)
    , m_HoverEnabled(true)
{
}

void VertexHighlighter::AddSelectedVertex(int vertexIndex) {
    if (!IsVertexSelected(vertexIndex)) {
        m_SelectedVertices.push_back(vertexIndex);
    }
}

void VertexHighlighter::RemoveSelectedVertex(int vertexIndex) {
    auto it = std::find(m_SelectedVertices.begin(), m_SelectedVertices.end(), vertexIndex);
    if (it != m_SelectedVertices.end()) {
        m_SelectedVertices.erase(it);
    }
}

bool VertexHighlighter::IsVertexSelected(int vertexIndex) const {
    return std::find(m_SelectedVertices.begin(), m_SelectedVertices.end(), vertexIndex) 
           != m_SelectedVertices.end();
}

void VertexHighlighter::ClearSelection() {
    m_SelectedVertices.clear();
}

void VertexHighlighter::Render(PhysXSoftBody* softBody) {
    if (!softBody) return;
    
    // Get vertex positions
    int vertexCount = softBody->GetVertexCount();
    std::vector<Vec3> positions(vertexCount);
    softBody->GetVertexPositions(positions.data());
    
    // Render selected vertices (green)
    Vec4 selectedColor(0.3f, 0.8f, 0.3f, 1.0f);
    for (int vertexIndex : m_SelectedVertices) {
        if (vertexIndex >= 0 && vertexIndex < vertexCount) {
            RenderVertex(positions[vertexIndex], selectedColor, 8.0f);
        }
    }
    
    // Render hovered vertex (yellow, larger)
    if (m_HoverEnabled && m_HoveredVertex >= 0 && m_HoveredVertex < vertexCount) {
        Vec4 hoverColor(0.8f, 0.8f, 0.3f, 1.0f);
        RenderVertex(positions[m_HoveredVertex], hoverColor, 12.0f);
    }
}

void VertexHighlighter::RenderVertex(const Vec3& position, const Vec4& color, float size) {
    // Save OpenGL state
    glPushAttrib(GL_POINT_BIT | GL_CURRENT_BIT | GL_ENABLE_BIT);
    
    // Disable depth test so vertices are always visible
    glDisable(GL_DEPTH_TEST);
    
    // Set point size
    glPointSize(size);
    
    // Render point
    glBegin(GL_POINTS);
    glColor4f(color.x, color.y, color.z, color.w);
    glVertex3f(position.x, position.y, position.z);
    glEnd();
    
    // Restore OpenGL state
    glPopAttrib();
}
