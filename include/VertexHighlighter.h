#pragma once

#include "Math/Vec3.h"
#include "Math/Vec4.h"
#include <vector>

class PhysXSoftBody;

/**
 * @brief Provides visual feedback for vertex selection
 * 
 * Renders highlighted vertices for hover and selection states.
 */
class VertexHighlighter {
public:
    VertexHighlighter();
    
    /**
     * @brief Set hovered vertex (shown in yellow)
     */
    void SetHoveredVertex(int vertexIndex) { m_HoveredVertex = vertexIndex; }
    int GetHoveredVertex() const { return m_HoveredVertex; }
    
    /**
     * @brief Add vertex to selection (shown in green)
     */
    void AddSelectedVertex(int vertexIndex);
    
    /**
     * @brief Remove vertex from selection
     */
    void RemoveSelectedVertex(int vertexIndex);
    
    /**
     * @brief Check if vertex is selected
     */
    bool IsVertexSelected(int vertexIndex) const;
    
    /**
     * @brief Clear all selected vertices
     */
    void ClearSelection();
    
    /**
     * @brief Get all selected vertices
     */
    const std::vector<int>& GetSelectedVertices() const { return m_SelectedVertices; }
    
    /**
     * @brief Enable/disable hover highlighting
     */
    void SetHoverEnabled(bool enabled) { m_HoverEnabled = enabled; }
    bool IsHoverEnabled() const { return m_HoverEnabled; }
    
    /**
     * @brief Render highlighted vertices
     * @param softBody Soft body to render highlights for
     */
    void Render(PhysXSoftBody* softBody);
    
private:
    int m_HoveredVertex;
    std::vector<int> m_SelectedVertices;
    bool m_HoverEnabled;
    
    void RenderVertex(const Vec3& position, const Vec4& color, float size);
};
